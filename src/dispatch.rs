//! Encode NRD dispatches into a compute pass.

use rusty_nrd::{DispatchDesc, ResourceBinding, ResourceType};

use crate::WgpuNrdError;
use crate::pools::PoolTextures;
use crate::resources::UserResources;

/// Owned copy of one frame’s dispatch (NRD invalidates its internal pointers on the next API call).
pub struct OwnedDispatch {
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
    pub constant_buffer: Vec<u8>,
    pub constant_matches_previous: bool,
    pub resources: Vec<ResourceBinding>,
}

impl OwnedDispatch {
    /// Clone data out of a [`DispatchDesc`] so the NRD borrow can end.
    pub fn from_desc(d: &DispatchDesc<'_>) -> Self {
        Self {
            pipeline_index: d.pipelineIndex,
            grid_width: d.gridWidth,
            grid_height: d.gridHeight,
            constant_buffer: d.constant_buffer().to_vec(),
            constant_matches_previous: d.constantBufferDataMatchesPreviousDispatch,
            resources: d.resources().to_vec(),
        }
    }
}

/// Resolve a [`TextureView`](wgpu::TextureView) for one resource bind group slot.
pub fn resolve_resource_view<'a>(
    le: &wgpu::BindGroupLayoutEntry,
    rd: &rusty_nrd::ResourceBinding,
    pools: &'a PoolTextures,
    user: &'a UserResources,
) -> Result<&'a wgpu::TextureView, WgpuNrdError> {
    if let Some(v) = user.by_binding.get(&le.binding) {
        return Ok(v);
    }
    let ty_u32 = rd.resource_type as u32;
    if let Some((sample, storage)) = user.split_sampled_storage.get(&ty_u32) {
        match &le.ty {
            wgpu::BindingType::Texture { .. } => return Ok(sample),
            wgpu::BindingType::StorageTexture { .. } => return Ok(storage),
            _ => {}
        }
    }
    let ty = rd.resource_type;
    match ty {
        ResourceType::TransientPool | ResourceType::PermanentPool => pools
            .view_for_pool(ty, rd.index_in_pool)
            .ok_or(WgpuNrdError::MissingResource {
                resource_type: ty,
                index_in_pool: rd.index_in_pool,
            }),
        _ => user
            .by_type
            .get(&ty_u32)
            .ok_or(WgpuNrdError::MissingResource {
                resource_type: ty,
                index_in_pool: rd.index_in_pool,
            }),
    }
}

/// Build bind group for NRD resource set (group `resourcesSpaceIndex`).
pub fn create_bind_group_resources(
    device: &wgpu::Device,
    label: Option<&str>,
    layout: &wgpu::BindGroupLayout,
    layout_entries: &[wgpu::BindGroupLayoutEntry],
    resources: &[rusty_nrd::ResourceBinding],
    pools: &PoolTextures,
    user: &UserResources,
) -> Result<wgpu::BindGroup, WgpuNrdError> {
    if layout_entries.len() != resources.len() {
        return Err(WgpuNrdError::SpirvReflect(format!(
            "resource count mismatch: layout {} vs dispatch {}",
            layout_entries.len(),
            resources.len()
        )));
    }

    let mut entries = Vec::with_capacity(resources.len());
    for (le, rd) in layout_entries.iter().zip(resources.iter()) {
        let view = resolve_resource_view(le, rd, pools, user)?;
        entries.push(wgpu::BindGroupEntry {
            binding: le.binding,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }

    Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout,
        entries: &entries,
    }))
}

/// Build bind group for constant buffer + samplers (group `constantBufferAndSamplersSpaceIndex`).
pub fn create_bind_group_constants(
    device: &wgpu::Device,
    label: Option<&str>,
    layout: &wgpu::BindGroupLayout,
    layout_entries: &[wgpu::BindGroupLayoutEntry],
    constant_buffer: &wgpu::Buffer,
    constant_size: u64,
    samplers: &[wgpu::Sampler],
) -> Result<wgpu::BindGroup, WgpuNrdError> {
    let mut sampler_idx = 0usize;
    let mut out = Vec::with_capacity(layout_entries.len());

    for le in layout_entries {
        let res = match le.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: constant_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(constant_size),
            }),
            wgpu::BindingType::Sampler(_) => {
                let s = samplers
                    .get(sampler_idx)
                    .ok_or_else(|| WgpuNrdError::SpirvReflect("not enough samplers".into()))?;
                sampler_idx += 1;
                wgpu::BindingResource::Sampler(s)
            }
            _ => {
                return Err(WgpuNrdError::SpirvReflect(
                    "unexpected binding in constant/sampler group".into(),
                ));
            }
        };
        out.push(wgpu::BindGroupEntry {
            binding: le.binding,
            resource: res,
        });
    }

    Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout,
        entries: &out,
    }))
}

/// Upload constant buffer data and record one dispatch.
pub fn encode_dispatch(
    pass: &mut wgpu::ComputePass<'_>,
    queue: &wgpu::Queue,
    constant_buffer: &wgpu::Buffer,
    bind_group_constants: &wgpu::BindGroup,
    bind_group_resources: &wgpu::BindGroup,
    pipeline: &wgpu::ComputePipeline,
    const_space: u32,
    resources_space: u32,
    dispatch: &DispatchDesc<'_>,
) {
    let size = dispatch.constant_buffer().len() as u64;
    if size > 0 {
        queue.write_buffer(constant_buffer, 0, dispatch.constant_buffer());
    }

    pass.set_pipeline(pipeline);
    pass.set_bind_group(const_space, bind_group_constants, &[]);
    pass.set_bind_group(resources_space, bind_group_resources, &[]);
    pass.dispatch_workgroups(dispatch.gridWidth as u32, dispatch.gridHeight as u32, 1);
}
