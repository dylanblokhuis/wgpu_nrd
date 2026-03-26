//! wgpu integration for [NVIDIA NRD](https://github.com/NVIDIAGameWorks/RayTracingDenoiser) via [`nrd_sys`].
//!
//! Shaders are loaded with [`wgpu::Device::create_shader_module_passthrough`] (SPIR-V, DXIL, or precompiled `.metallib` on Metal from NRD).
//!
//! [`WgpuNrd::new`] builds the NRD [`Instance`] from [`DenoiserSlot`]s; NRD types used by typical apps are re-exported so you do not need a direct `nrd-sys` dependency.

#![allow(missing_docs)]

mod dispatch;
mod error;
mod format;
mod pipelines;
mod pools;
mod reflect;
mod resources;

pub use dispatch::{
    OwnedDispatch, create_bind_group_constants, create_bind_group_resources, encode_dispatch,
    resolve_resource_view,
};
pub use error::WgpuNrdError;
pub use format::{
    format_from_raw, nrd_format_to_wgpu, pool_extent, user_resource_storage_wgpu_format,
    wgpu_format_for_resource_binding, wgpu_texture_binding_sample_type,
};
pub use pipelines::{PipelineState, build_pipelines, clone_dispatch_resource_lists};
pub use pools::PoolTextures;
pub use reflect::{
    bind_group_layout_entries, compute_workgroup_size, parse_spirv, spirv_image_format_to_wgpu,
};
pub use resources::{UserResourceMap, UserResources, insert_user_resource};

use std::ffi::CStr;

/// Re-exports for NRD configuration without a direct `nrd-sys` dependency.
pub use nrd_sys::{
    Denoiser, DenoiserSlot, Identifier, Instance, LibraryInfo, ResourceType,
    default_common_settings, default_reblur_settings,
};

/// Owns NRD state, pool textures, samplers, constant buffer, and compute pipelines for one [`Instance`].
pub struct WgpuNrd {
    /// NRD instance.
    pub instance: Instance,
    /// Linked library metadata.
    pub library: LibraryInfo,
    /// GPU pools (permanent + transient).
    pub pools: PoolTextures,
    /// wgpu sampler objects matching [`InstanceDescription::samplers`].
    pub samplers: Vec<wgpu::Sampler>,
    /// Uniform scratch for NRD constants (`constantBufferMaxDataSize` bytes).
    pub constant_buffer: wgpu::Buffer,
    /// Bind group layout for constant buffer + samplers (`constantBufferAndSamplersSpaceIndex`).
    pub bind_group_layout_constants: wgpu::BindGroupLayout,
    /// Layout entries for [`bind_group_layout_constants`] (SPIR-V reflection).
    pub constant_layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// One entry per NRD pipeline index.
    pub pipelines: Vec<PipelineState>,
    /// Compute entry name (e.g. `NRD_CS_MAIN`).
    pub shader_entry: String,
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_max_data_size: u32,
    backend: wgpu::Backend,
}

impl WgpuNrd {
    /// Creates an NRD [`Instance`] from `denoisers`, then builds GPU resources. `backend` must match the device (see [`wgpu::AdapterInfo::backend`]).
    ///
    /// `compute_dispatch_identifiers` is passed to NRD [`Instance::compute_dispatches`] so storage texture
    /// formats in bind group layouts match pool textures and user inputs (SPIR-V often uses
    /// `StorageImageWriteWithoutFormat`). Use the same [`Identifier`] values as when you call
    /// [`WgpuNrd::encode_dispatches`]. An empty slice skips that patch (SPIR-V reflection defaults only).
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        denoisers: &[DenoiserSlot],
        backend: wgpu::Backend,
        resource_width: u32,
        resource_height: u32,
        compute_dispatch_identifiers: &[Identifier],
    ) -> Result<Self, WgpuNrdError> {
        let mut instance = Instance::try_new_denoisers(denoisers)?;
        let library = LibraryInfo::query()?;

        let (
            shader_entry,
            pools,
            samplers,
            constant_buffer,
            pipelines,
            bind_group_layout_constants,
            constant_layout_entries,
            constant_buffer_and_samplers_space_index,
            resources_space_index,
            constant_buffer_max_data_size,
        ) = {
            let dispatch_resources =
                clone_dispatch_resource_lists(&mut instance, compute_dispatch_identifiers)?;
            let desc = instance.description()?;
            let raw = desc.raw();

            let shader_entry = if raw.shaderEntryPoint.is_null() {
                "NRD_CS_MAIN".to_string()
            } else {
                unsafe { CStr::from_ptr(raw.shaderEntryPoint) }
                    .to_str()
                    .map_err(|e| WgpuNrdError::SpirvReflect(e.to_string()))?
                    .to_string()
            };

            let permanent = desc.permanent_pool().to_vec();
            let transient = desc.transient_pool().to_vec();

            let pools = PoolTextures::new(
                device,
                adapter,
                Some("nrd_pool"),
                &permanent,
                &transient,
                resource_width,
                resource_height,
            )?;

            let samplers: Vec<wgpu::Sampler> = desc
                .samplers()
                .iter()
                .map(|&s| device.create_sampler(&sampler_desc_from_nrd(s)))
                .collect();

            let max_cb = raw.constantBufferMaxDataSize.max(1) as u64;
            let constant_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nrd_constants"),
                size: max_cb,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let pipelines_slice = desc.pipelines();
            if pipelines_slice.is_empty() {
                return Err(WgpuNrdError::SpirvReflect(
                    "no pipelines in NRD instance".into(),
                ));
            }

            let (pipelines, bind_group_layout_constants, constant_layout_entries) =
                build_pipelines(
                    device,
                    backend,
                    raw,
                    pipelines_slice,
                    &dispatch_resources,
                    &permanent,
                    &transient,
                )?;

            (
                shader_entry,
                pools,
                samplers,
                constant_buffer,
                pipelines,
                bind_group_layout_constants,
                constant_layout_entries,
                raw.constantBufferAndSamplersSpaceIndex,
                raw.resourcesSpaceIndex,
                raw.constantBufferMaxDataSize,
            )
        };

        Ok(Self {
            instance,
            library,
            pools,
            samplers,
            constant_buffer,
            bind_group_layout_constants,
            constant_layout_entries,
            pipelines,
            shader_entry,
            constant_buffer_and_samplers_space_index,
            resources_space_index,
            constant_buffer_max_data_size,
            backend,
        })
    }

    /// Encode all dispatches for the given identifiers (after updating NRD settings on [`Self::instance`]).
    ///
    /// Each NRD dispatch runs in its **own** compute pass so texture usage does not span multiple
    /// `dispatch_workgroups` calls (avoids STORAGE vs SAMPLE conflicts on the same texture across
    /// successive pipelines in one pass).
    pub fn encode_dispatches(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        identifiers: &[Identifier],
        user: &UserResources,
    ) -> Result<(), WgpuNrdError> {
        let owned: Vec<OwnedDispatch> = {
            let dispatches = self.instance.compute_dispatches(identifiers)?;
            dispatches.iter().map(OwnedDispatch::from_desc).collect()
        };
        for d in &owned {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.encode_one_owned(device, queue, &mut pass, d, user)?;
        }
        Ok(())
    }

    /// Single dispatch from owned data (see [`OwnedDispatch`]).
    pub fn encode_one_owned(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::ComputePass<'_>,
        dispatch: &OwnedDispatch,
        user: &UserResources,
    ) -> Result<(), WgpuNrdError> {
        let idx = dispatch.pipeline_index as usize;
        let ps = self
            .pipelines
            .get(idx)
            .ok_or_else(|| WgpuNrdError::SpirvReflect(format!("bad pipeline index {idx}")))?;

        let resources: &[nrd_sys::ResourceBinding] = &dispatch.resources;
        let bg0 = create_bind_group_constants(
            device,
            Some("nrd_bg_constants"),
            &self.bind_group_layout_constants,
            &self.constant_layout_entries,
            &self.constant_buffer,
            self.constant_buffer_max_data_size as u64,
            &self.samplers,
        )?;
        let bg1 = create_bind_group_resources(
            device,
            Some("nrd_bg_resources"),
            &ps.bind_group_layout_resources,
            &ps.resource_entries,
            resources,
            &self.pools,
            user,
        )?;

        let cb_size = dispatch.constant_buffer.len() as u64;
        if cb_size > self.constant_buffer_max_data_size as u64 {
            return Err(WgpuNrdError::SpirvReflect(format!(
                "constant buffer size {} exceeds max {}",
                cb_size, self.constant_buffer_max_data_size
            )));
        }
        if !dispatch.constant_matches_previous && cb_size > 0 {
            queue.write_buffer(&self.constant_buffer, 0, &dispatch.constant_buffer);
        }

        pass.set_pipeline(&ps.compute_pipeline);
        pass.set_bind_group(self.constant_buffer_and_samplers_space_index, &bg0, &[]);
        pass.set_bind_group(self.resources_space_index, &bg1, &[]);
        pass.dispatch_workgroups(dispatch.grid_width as u32, dispatch.grid_height as u32, 1);
        Ok(())
    }

    /// Active graphics backend.
    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }
}

fn sampler_desc_from_nrd(s: nrd_sys::Sampler) -> wgpu::SamplerDescriptor<'static> {
    use nrd_sys::Sampler::*;
    let (mag, min) = match s {
        NearestClamp => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest),
        LinearClamp => (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear),
        MaxNum => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest),
    };
    wgpu::SamplerDescriptor {
        label: Some("nrd"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    }
}
