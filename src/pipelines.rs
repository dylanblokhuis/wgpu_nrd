//! Shader modules (passthrough) and compute pipeline cache.

use std::borrow::Cow;
use std::ffi::CStr;

use crate::WgpuNrdError;
use crate::format::{wgpu_format_for_resource_binding, wgpu_texture_binding_sample_type};
use crate::reflect::{bind_group_layout_entries, compute_workgroup_size, parse_spirv};
use nrd_sys::ffi;
use nrd_sys::{DispatchDesc, Identifier, Instance, ResourceBinding};

/// One NRD pipeline: shader + layouts + compute pipeline.
pub struct PipelineState {
    /// Layout entries for bind group `resourcesSpaceIndex` (sorted by binding), for dispatch binding.
    pub resource_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub bind_group_layout_resources: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub shader_module: wgpu::ShaderModule,
}

/// SPIR-V slice from [`ffi::nrd_ComputeShaderDesc`].
fn spirv_words(desc: &ffi::nrd_ComputeShaderDesc) -> Result<&[u32], WgpuNrdError> {
    if desc.size == 0 || desc.bytecode.is_null() {
        return Err(WgpuNrdError::InvalidSpirvSize(0));
    }
    let len = desc.size as usize;
    if len % 4 != 0 {
        return Err(WgpuNrdError::InvalidSpirvSize(len));
    }
    let words = len / 4;
    Ok(unsafe { std::slice::from_raw_parts(desc.bytecode as *const u32, words) })
}

fn spirv_bytes(desc: &ffi::nrd_ComputeShaderDesc) -> Result<&[u8], WgpuNrdError> {
    if desc.size == 0 || desc.bytecode.is_null() {
        return Err(WgpuNrdError::InvalidSpirvSize(0));
    }
    let len = desc.size as usize;
    if len % 4 != 0 {
        return Err(WgpuNrdError::InvalidSpirvSize(len));
    }
    Ok(unsafe { std::slice::from_raw_parts(desc.bytecode as *const u8, len) })
}

fn dxil_bytes(desc: &ffi::nrd_ComputeShaderDesc) -> &[u8] {
    if desc.size == 0 || desc.bytecode.is_null() {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(desc.bytecode as *const u8, desc.size as usize) }
}

fn metal_lib_bytes(desc: &ffi::nrd_ComputeShaderDesc) -> &[u8] {
    if desc.size == 0 || desc.bytecode.is_null() {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(desc.bytecode as *const u8, desc.size as usize) }
}

/// Create passthrough shader module for the active backend (SPIR-V, DXIL, or precompiled `.metallib` on Metal).
pub unsafe fn shader_module_passthrough(
    device: &wgpu::Device,
    backend: wgpu::Backend,
    pipeline: &ffi::nrd_PipelineDesc,
    workgroup: (u32, u32, u32),
    label: Option<&str>,
) -> Result<wgpu::ShaderModule, WgpuNrdError> {
    let mut desc = wgpu::ShaderModuleDescriptorPassthrough::default();
    desc.label = label;
    desc.num_workgroups = workgroup;

    match backend {
        wgpu::Backend::Vulkan
        | wgpu::Backend::Gl
        | wgpu::Backend::Noop
        | wgpu::Backend::BrowserWebGpu => {
            let words = spirv_words(&pipeline.computeShaderSPIRV)?;
            desc.spirv = Some(Cow::Borrowed(words));
        }
        wgpu::Backend::Dx12 => {
            let dxil = dxil_bytes(&pipeline.computeShaderDXIL);
            if dxil.is_empty() {
                return Err(WgpuNrdError::SpirvReflect(
                    "empty DXIL in nrd_PipelineDesc".into(),
                ));
            }
            desc.dxil = Some(Cow::Borrowed(dxil));
        }
        wgpu::Backend::Metal => {
            let metallib = metal_lib_bytes(&pipeline.computeShaderMetal);
            if metallib.is_empty() {
                return Err(WgpuNrdError::SpirvReflect(
                    "empty Metal metallib in nrd_PipelineDesc.computeShaderMetal".into(),
                ));
            }
            desc.metallib = Some(Cow::Borrowed(metallib));
        }
    }

    Ok(unsafe { device.create_shader_module_passthrough(desc) })
}

/// Snapshot resource bindings from [`Instance::compute_dispatches`] for storage format patching.
/// Call this before other [`Instance`] borrows (e.g. [`nrd_sys::Instance::description`]) so the
/// instance can be mutably borrowed here.
pub fn clone_dispatch_resource_lists(
    instance: &mut Instance,
    ids: &[Identifier],
) -> Result<Vec<(u16, Vec<ResourceBinding>)>, WgpuNrdError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let dispatches = instance
        .compute_dispatches(ids)
        .map_err(WgpuNrdError::Nrd)?;
    Ok(dispatches
        .iter()
        .filter_map(|d: &DispatchDesc<'_>| {
            let r = d.resources();
            if r.is_empty() {
                None
            } else {
                Some((d.pipelineIndex, r.to_vec()))
            }
        })
        .collect())
}

fn patch_resource_layout_from_nrd(
    entries: &mut [wgpu::BindGroupLayoutEntry],
    resources: &[ResourceBinding],
    permanent: &[ffi::nrd_TextureDesc],
    transient: &[ffi::nrd_TextureDesc],
) -> Result<(), WgpuNrdError> {
    if resources.len() != entries.len() {
        return Ok(());
    }
    for (entry, rd) in entries.iter_mut().zip(resources.iter()) {
        let Some(fmt) = wgpu_format_for_resource_binding(rd, permanent, transient)? else {
            continue;
        };
        match &mut entry.ty {
            wgpu::BindingType::StorageTexture { format, .. } => {
                *format = fmt;
            }
            wgpu::BindingType::Texture { sample_type, .. } => {
                if let Some(st) = wgpu_texture_binding_sample_type(fmt) {
                    *sample_type = st;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Build all pipeline states for an NRD instance description.
///
/// Returns pipeline states, bind group layout for constants/samplers, and its reflected entries.
pub fn build_pipelines(
    device: &wgpu::Device,
    backend: wgpu::Backend,
    raw: &ffi::nrd_InstanceDesc,
    pipelines: &[ffi::nrd_PipelineDesc],
    dispatch_resources: &[(u16, Vec<ResourceBinding>)],
    permanent: &[ffi::nrd_TextureDesc],
    transient: &[ffi::nrd_TextureDesc],
) -> Result<
    (
        Vec<PipelineState>,
        wgpu::BindGroupLayout,
        Vec<wgpu::BindGroupLayoutEntry>,
    ),
    WgpuNrdError,
> {
    let entry_cstr = if raw.shaderEntryPoint.is_null() {
        "main"
    } else {
        unsafe { CStr::from_ptr(raw.shaderEntryPoint) }
            .to_str()
            .map_err(|e| WgpuNrdError::SpirvReflect(e.to_string()))?
    };
    let entry = entry_cstr.to_string();

    if pipelines.is_empty() {
        return Err(WgpuNrdError::SpirvReflect("no NRD pipelines".into()));
    }

    // Reflect set 0 from first pipeline SPIR-V
    let spirv0 = spirv_bytes(&pipelines[0].computeShaderSPIRV)?;
    let reflect0 = parse_spirv(spirv0)?;
    let entries0 = bind_group_layout_entries(
        &reflect0,
        raw.constantBufferAndSamplersSpaceIndex,
        raw.constantBufferMaxDataSize,
    )?;
    let bind_group_layout_cb = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nrd_set_constant_samplers"),
        entries: &entries0,
    });
    let entries0_owned = entries0.clone();

    let mut out = Vec::with_capacity(pipelines.len());

    for (i, p) in pipelines.iter().enumerate() {
        let spirv = spirv_bytes(&p.computeShaderSPIRV)?;
        let reflection = parse_spirv(spirv)?;
        let wg = compute_workgroup_size(&reflection, &entry)?;

        let mut entries1 = bind_group_layout_entries(&reflection, raw.resourcesSpaceIndex, 1)?;
        if let Some((_, resources)) = dispatch_resources.iter().find(|(pid, _)| *pid == i as u16) {
            patch_resource_layout_from_nrd(&mut entries1, resources, permanent, transient)?;
        }
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("nrd_set_resources_{i}")),
            entries: &entries1,
        });

        let s_cb = raw.constantBufferAndSamplersSpaceIndex;
        let s_res = raw.resourcesSpaceIndex;
        let max_g = s_cb.max(s_res);
        let mut layouts_ordered: Vec<&wgpu::BindGroupLayout> = Vec::new();
        for g in 0..=max_g {
            if g == s_cb {
                layouts_ordered.push(&bind_group_layout_cb);
            } else if g == s_res {
                layouts_ordered.push(&bgl1);
            } else {
                return Err(WgpuNrdError::SpirvReflect(format!(
                    "NRD descriptor space gap at group {g} (expected only {s_cb} and {s_res})"
                )));
            }
        }

        let bgl_opts: Vec<Option<&wgpu::BindGroupLayout>> =
            layouts_ordered.into_iter().map(Some).collect();
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("nrd_pipeline_layout_{i}")),
            bind_group_layouts: &bgl_opts,
            immediate_size: 0,
        });

        let sm = unsafe { shader_module_passthrough(device, backend, p, wg, Some("nrd_shader"))? };

        let entry = if entry == "main" && backend == wgpu::Backend::Metal {
            "main0"
        } else {
            entry.as_str()
        };

        let cp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("nrd_compute_{i}")),
            layout: Some(&pl),
            module: &sm,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });

        out.push(PipelineState {
            resource_entries: entries1,
            bind_group_layout_resources: bgl1,
            pipeline_layout: pl,
            compute_pipeline: cp,
            shader_module: sm,
        });
    }

    Ok((out, bind_group_layout_cb, entries0_owned))
}
