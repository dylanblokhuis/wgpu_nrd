//! Permanent / transient pool textures.

use crate::WgpuNrdError;
use crate::format::{format_from_raw, nrd_format_to_wgpu, pool_extent};
use rusty_nrd::ffi;

/// GPU textures backing NRD permanent and transient pools (indexed by `index_in_pool` in dispatches).
pub struct PoolTextures {
    /// `permanentPool[i]` textures and views.
    pub permanent: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// `transientPool[i]` textures and views.
    pub transient: Vec<(wgpu::Texture, wgpu::TextureView)>,
}

impl PoolTextures {
    /// Allocate all pool textures for the given resource size (from common settings).
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        label_prefix: Option<&str>,
        permanent: &[ffi::nrd_TextureDesc],
        transient: &[ffi::nrd_TextureDesc],
        resource_width: u32,
        resource_height: u32,
    ) -> Result<Self, WgpuNrdError> {
        let alloc = |desc: &ffi::nrd_TextureDesc,
                     idx: usize,
                     pool: &str|
         -> Result<(wgpu::Texture, wgpu::TextureView), WgpuNrdError> {
            let fmt = format_from_raw(desc.format)?;
            let wgpu_fmt = nrd_format_to_wgpu(fmt)?;
            let extent = pool_extent(resource_width, resource_height, desc.downsampleFactor);
            let label = label_prefix.map(|p| format!("{p}_{pool}_{idx}"));
            let desired =
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
            let allowed = adapter.get_texture_format_features(wgpu_fmt).allowed_usages;
            let usage = desired & allowed;
            if usage.is_empty() {
                return Err(WgpuNrdError::PoolTextureFormatNotBindable {
                    format: wgpu_fmt,
                    allowed,
                });
            }
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_fmt,
                usage,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Ok((tex, view))
        };

        let mut permanent_v = Vec::with_capacity(permanent.len());
        for (i, d) in permanent.iter().enumerate() {
            permanent_v.push(alloc(d, i, "perm")?);
        }
        let mut transient_v = Vec::with_capacity(transient.len());
        for (i, d) in transient.iter().enumerate() {
            transient_v.push(alloc(d, i, "trans")?);
        }
        Ok(Self {
            permanent: permanent_v,
            transient: transient_v,
        })
    }

    /// Resolve a pool `index_in_pool` for [`rusty_nrd::ResourceType::TransientPool`] or [`PermanentPool`](rusty_nrd::ResourceType::PermanentPool).
    pub fn view_for_pool(
        &self,
        pool: rusty_nrd::ResourceType,
        index_in_pool: u16,
    ) -> Option<&wgpu::TextureView> {
        let idx = index_in_pool as usize;
        match pool {
            rusty_nrd::ResourceType::TransientPool => self.transient.get(idx).map(|(_, v)| v),
            rusty_nrd::ResourceType::PermanentPool => self.permanent.get(idx).map(|(_, v)| v),
            _ => None,
        }
    }
}
