//! Shader modules (passthrough) and compute pipeline cache.

use std::borrow::Cow;
use std::ffi::CStr;

use crate::WgpuNrdError;
use crate::format::{wgpu_format_for_resource_binding, wgpu_texture_binding_sample_type};
use crate::reflect::rspirv_reflect::DescriptorType as ReflectDescriptorType;
use crate::reflect::rspirv_reflect::rspirv::binary::{Assemble, Parser};
use crate::reflect::rspirv_reflect::rspirv::dr::{Loader, Module, Operand};
use crate::reflect::rspirv_reflect::spirv;
use crate::reflect::{bind_group_layout_entries, compute_workgroup_size, parse_spirv};
use rusty_nrd::ffi;
use rusty_nrd::{DispatchDesc, Identifier, Instance, ResourceBinding};

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
fn spirv_words(desc: &ffi::nrd_ComputeShaderDesc) -> Result<Vec<u32>, WgpuNrdError> {
    let bytes = spirv_bytes(desc)?;
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

fn spirv_words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
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

fn parse_module(words: &[u32]) -> Result<Module, WgpuNrdError> {
    let bytes = spirv_words_to_bytes(words);
    let mut loader = Loader::new();
    Parser::new(&bytes, &mut loader)
        .parse()
        .map_err(|e| WgpuNrdError::SpirvReflect(e.to_string()))?;
    Ok(loader.module())
}

fn find_assignment(
    module: &Module,
    id: u32,
) -> Option<&crate::reflect::rspirv_reflect::rspirv::dr::Instruction> {
    module
        .types_global_values
        .iter()
        .find(|i| i.result_id == Some(id))
}

fn operand_id_ref(
    inst: &crate::reflect::rspirv_reflect::rspirv::dr::Instruction,
    idx: usize,
) -> Result<u32, WgpuNrdError> {
    match inst.operands.get(idx) {
        Some(Operand::IdRef(id)) => Ok(*id),
        _ => Err(WgpuNrdError::SpirvReflect(format!(
            "operand {idx} not IdRef for {:?}",
            inst.class.opcode
        ))),
    }
}

fn resolve_type_image<'a>(
    module: &'a Module,
    mut type_id: u32,
) -> Result<&'a crate::reflect::rspirv_reflect::rspirv::dr::Instruction, WgpuNrdError> {
    loop {
        let ty = find_assignment(module, type_id).ok_or_else(|| {
            WgpuNrdError::SpirvReflect(format!("missing SPIR-V type id {type_id}"))
        })?;
        match ty.class.opcode {
            spirv::Op::TypePointer => type_id = operand_id_ref(ty, 1)?,
            spirv::Op::TypeArray | spirv::Op::TypeRuntimeArray => type_id = operand_id_ref(ty, 0)?,
            spirv::Op::TypeSampledImage => type_id = operand_id_ref(ty, 0)?,
            spirv::Op::TypeImage => return Ok(ty),
            _ => {
                return Err(WgpuNrdError::SpirvReflect(format!(
                    "expected image-related type, got {:?}",
                    ty.class.opcode
                )));
            }
        }
    }
}

fn operand_sampled(inst: &crate::reflect::rspirv_reflect::rspirv::dr::Instruction) -> u32 {
    match inst.operands.get(5) {
        Some(Operand::LiteralBit32(v)) => *v,
        _ => 1,
    }
}

fn set_binding_for_var(module: &Module, var_id: u32) -> (Option<u32>, Option<u32>) {
    let mut set = None;
    let mut binding = None;
    for ann in module
        .annotations
        .iter()
        .filter(|a| a.class.opcode == spirv::Op::Decorate)
    {
        if ann.operands.first() != Some(&Operand::IdRef(var_id)) {
            continue;
        }
        if let (
            Some(Operand::Decoration(spirv::Decoration::DescriptorSet)),
            Some(Operand::LiteralBit32(s)),
        ) = (ann.operands.get(1), ann.operands.get(2))
        {
            set = Some(*s);
        }
        if let (
            Some(Operand::Decoration(spirv::Decoration::Binding)),
            Some(Operand::LiteralBit32(b)),
        ) = (ann.operands.get(1), ann.operands.get(2))
        {
            binding = Some(*b);
        }
    }
    (set, binding)
}

fn collect_set_image_bindings(
    module: &Module,
    set_index: u32,
) -> Result<Vec<(u32, ffi::nrd_DescriptorType)>, WgpuNrdError> {
    let mut out: Vec<(u32, ffi::nrd_DescriptorType)> = Vec::new();
    for var in module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == spirv::Op::Variable)
    {
        let var_id = match var.result_id {
            Some(v) => v,
            None => continue,
        };
        let (set, binding) = set_binding_for_var(module, var_id);
        if set != Some(set_index) {
            continue;
        }
        let Some(binding) = binding else { continue };
        let Some(type_id) = var.result_type else {
            continue;
        };
        let image_ty = match resolve_type_image(module, type_id) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let nrd_ty = if operand_sampled(image_ty) == 2 {
            ffi::nrd_DescriptorType_STORAGE_TEXTURE
        } else {
            ffi::nrd_DescriptorType_TEXTURE
        };
        out.push((binding, nrd_ty));
    }
    out.sort_by_key(|(b, _)| *b);
    out.dedup_by_key(|(b, _)| *b);
    Ok(out)
}

fn resource_range_descriptor_types(
    pipeline: &ffi::nrd_PipelineDesc,
) -> Vec<ffi::nrd_DescriptorType> {
    let mut out = Vec::new();
    if pipeline.resourceRanges.is_null() || pipeline.resourceRangesNum == 0 {
        return out;
    }
    let ranges = unsafe {
        std::slice::from_raw_parts(pipeline.resourceRanges, pipeline.resourceRangesNum as usize)
    };
    for r in ranges {
        for _ in 0..r.descriptorsNum {
            out.push(r.descriptorType);
        }
    }
    out
}

fn remap_set_bindings_dense(
    words: &[u32],
    pipeline: &ffi::nrd_PipelineDesc,
    set_index: u32,
    base_binding: u32,
) -> Result<Vec<u32>, WgpuNrdError> {
    let mut module = parse_module(words)?;
    let declared = collect_set_image_bindings(&module, set_index)?;
    if declared.is_empty() {
        return Ok(words.to_vec());
    }

    let expected = resource_range_descriptor_types(pipeline);
    if expected.is_empty() {
        return Ok(words.to_vec());
    }

    let mut by_kind_tex: Vec<u32> = declared
        .iter()
        .filter_map(|(b, t)| {
            if *t == ffi::nrd_DescriptorType_TEXTURE {
                Some(*b)
            } else {
                None
            }
        })
        .collect();
    let mut by_kind_stor: Vec<u32> = declared
        .iter()
        .filter_map(|(b, t)| {
            if *t == ffi::nrd_DescriptorType_STORAGE_TEXTURE {
                Some(*b)
            } else {
                None
            }
        })
        .collect();
    let mut any_remaining: Vec<u32> = declared.iter().map(|(b, _)| *b).collect();

    let mut new_order: Vec<u32> = Vec::new();
    for want in expected {
        let picked = if want == ffi::nrd_DescriptorType_STORAGE_TEXTURE {
            by_kind_stor
                .iter()
                .position(|b| any_remaining.contains(b))
                .map(|idx| by_kind_stor.remove(idx))
        } else {
            by_kind_tex
                .iter()
                .position(|b| any_remaining.contains(b))
                .map(|idx| by_kind_tex.remove(idx))
        }
        .or_else(|| any_remaining.first().copied());
        let Some(old_binding) = picked else { break };
        any_remaining.retain(|b| *b != old_binding);
        new_order.push(old_binding);
    }
    new_order.extend(any_remaining.iter().copied());

    let mut mapping = std::collections::BTreeMap::new();
    for (i, old) in new_order.iter().enumerate() {
        mapping.insert(*old, base_binding + i as u32);
    }

    // Rewrite OpDecorate Binding for variables in the target set.
    let mut target_var_ids = std::collections::BTreeSet::new();
    for var in module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == spirv::Op::Variable)
    {
        let Some(var_id) = var.result_id else {
            continue;
        };
        let (set, _) = set_binding_for_var(&module, var_id);
        if set == Some(set_index) {
            target_var_ids.insert(var_id);
        }
    }

    for ann in module
        .annotations
        .iter_mut()
        .filter(|a| a.class.opcode == spirv::Op::Decorate)
    {
        let var_id = match ann.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        if !target_var_ids.contains(&var_id) {
            continue;
        }
        let is_binding = matches!(
            ann.operands.get(1),
            Some(Operand::Decoration(spirv::Decoration::Binding))
        );
        if !is_binding {
            continue;
        }
        let old = match ann.operands.get(2) {
            Some(Operand::LiteralBit32(v)) => *v,
            _ => continue,
        };
        if let Some(new_b) = mapping.get(&old).copied() {
            ann.operands[2] = Operand::LiteralBit32(new_b);
        }
    }

    Ok(module.assemble())
}

/// Create passthrough shader module for the active backend (SPIR-V, DXIL, or precompiled `.metallib` on Metal).
pub unsafe fn shader_module_passthrough(
    device: &wgpu::Device,
    backend: wgpu::Backend,
    pipeline: &ffi::nrd_PipelineDesc,
    remapped_spirv_words: Option<Vec<u32>>,
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
            let words = if let Some(words) = remapped_spirv_words {
                words
            } else {
                spirv_words(&pipeline.computeShaderSPIRV)?
            };
            desc.spirv = Some(Cow::Owned(words));
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
/// Call this before other [`Instance`] borrows (e.g. [`rusty_nrd::Instance::description`]) so the
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

fn infer_space_indices(
    raw: &ffi::nrd_InstanceDesc,
    pipelines: &[ffi::nrd_PipelineDesc],
) -> Result<(u32, u32), WgpuNrdError> {
    let Some(first) = pipelines.first() else {
        return Err(WgpuNrdError::SpirvReflect("no NRD pipelines".into()));
    };
    let reflection = parse_spirv(spirv_bytes(&first.computeShaderSPIRV)?)?;
    let sets = reflection
        .get_descriptor_sets()
        .map_err(|e| WgpuNrdError::SpirvReflect(e.to_string()))?;

    // Prefer explicit register hints from NRD instance description when they resolve unambiguously.
    let cb_reg = raw.constantBufferRegisterIndex;
    let res_reg = raw.resourcesBaseRegisterIndex;
    let cb_from_reg = sets.iter().find_map(|(&set, bindings)| {
        let b = bindings.get(&cb_reg)?;
        if matches!(
            b.ty,
            ReflectDescriptorType::UNIFORM_BUFFER
                | ReflectDescriptorType::UNIFORM_BUFFER_DYNAMIC
                | ReflectDescriptorType::INLINE_UNIFORM_BLOCK_EXT
        ) {
            Some(set)
        } else {
            None
        }
    });
    let res_from_reg = sets.iter().find_map(|(&set, bindings)| {
        let b = bindings.get(&res_reg)?;
        if matches!(
            b.ty,
            ReflectDescriptorType::SAMPLED_IMAGE
                | ReflectDescriptorType::STORAGE_IMAGE
                | ReflectDescriptorType::COMBINED_IMAGE_SAMPLER
        ) {
            Some(set)
        } else {
            None
        }
    });
    if let (Some(cb), Some(res)) = (cb_from_reg, res_from_reg) {
        if cb != res {
            return Ok((cb, res));
        }
    }

    // Fallback heuristic: constants set has UBO/samplers; resources set has sampled/storage images.
    let mut cb_candidates = Vec::new();
    let mut res_candidates = Vec::new();
    for (&set, bindings) in &sets {
        let mut has_uniform = false;
        let mut has_sampler = false;
        let mut has_image = false;
        for info in bindings.values() {
            match info.ty {
                ReflectDescriptorType::UNIFORM_BUFFER
                | ReflectDescriptorType::UNIFORM_BUFFER_DYNAMIC
                | ReflectDescriptorType::INLINE_UNIFORM_BLOCK_EXT => has_uniform = true,
                ReflectDescriptorType::SAMPLER => has_sampler = true,
                ReflectDescriptorType::SAMPLED_IMAGE
                | ReflectDescriptorType::STORAGE_IMAGE
                | ReflectDescriptorType::COMBINED_IMAGE_SAMPLER => has_image = true,
                _ => {}
            }
        }
        if has_uniform || has_sampler {
            cb_candidates.push(set);
        }
        if has_image {
            res_candidates.push(set);
        }
    }
    cb_candidates.sort_unstable();
    cb_candidates.dedup();
    res_candidates.sort_unstable();
    res_candidates.dedup();

    if cb_candidates.len() == 1
        && res_candidates.len() == 1
        && cb_candidates[0] != res_candidates[0]
    {
        return Ok((cb_candidates[0], res_candidates[0]));
    }

    Ok((
        raw.constantBufferAndSamplersSpaceIndex,
        raw.resourcesSpaceIndex,
    ))
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
        u32,
        u32,
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
    let (s_cb, s_res) = infer_space_indices(raw, pipelines)?;
    let remapped_words: Vec<Vec<u32>> = pipelines
        .iter()
        .map(|p| {
            let words = spirv_words(&p.computeShaderSPIRV)?;
            remap_set_bindings_dense(&words, p, s_res, raw.resourcesBaseRegisterIndex)
        })
        .collect::<Result<_, _>>()?;

    // Constant + sampler set: union bindings from every pipeline (each pass may add its own).
    // Reflecting only pipelines[0] misses pass-specific UBOs and fails Vulkan layout validation.
    let mut entries0_map: std::collections::BTreeMap<u32, wgpu::BindGroupLayoutEntry> =
        std::collections::BTreeMap::new();
    for words in &remapped_words {
        let spirv = spirv_words_to_bytes(words);
        let reflection = parse_spirv(&spirv)?;
        let entries = bind_group_layout_entries(&reflection, s_cb, raw.constantBufferMaxDataSize)?;
        for e in entries {
            entries0_map.entry(e.binding).or_insert(e);
        }
    }
    let entries0: Vec<wgpu::BindGroupLayoutEntry> = entries0_map.into_values().collect();
    let bind_group_layout_cb = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nrd_set_constant_samplers"),
        entries: &entries0,
    });
    let entries0_owned = entries0.clone();

    let mut out = Vec::with_capacity(pipelines.len());

    for (i, p) in pipelines.iter().enumerate() {
        let spirv = spirv_words_to_bytes(&remapped_words[i]);
        let reflection = parse_spirv(&spirv)?;
        let wg = compute_workgroup_size(&reflection, &entry)?;

        let mut entries1 = bind_group_layout_entries(&reflection, s_res, 1)?;
        if let Some((_, resources)) = dispatch_resources.iter().find(|(pid, _)| *pid == i as u16) {
            patch_resource_layout_from_nrd(&mut entries1, resources, permanent, transient)?;
        }
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("nrd_set_resources_{i}")),
            entries: &entries1,
        });

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

        let sm = unsafe {
            shader_module_passthrough(
                device,
                backend,
                p,
                Some(remapped_words[i].clone()),
                wg,
                Some("nrd_shader"),
            )?
        };

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

    Ok((out, bind_group_layout_cb, entries0_owned, s_cb, s_res))
}
