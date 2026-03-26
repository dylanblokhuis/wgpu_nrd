//! `nrd_sys::Format` ↔ `wgpu::TextureFormat` and extent helpers.

use crate::WgpuNrdError;
use nrd_sys::{ffi, Format, ResourceType};

/// Convert NRD format to wgpu texture format.
pub fn nrd_format_to_wgpu(f: Format) -> Result<wgpu::TextureFormat, WgpuNrdError> {
    use Format::*;
    use wgpu::TextureFormat as T;
    let out = match f {
        R8Unorm => T::R8Unorm,
        R8Snorm => T::R8Snorm,
        R8Uint => T::R8Uint,
        R8Sint => T::R8Sint,
        Rg8Unorm => T::Rg8Unorm,
        Rg8Snorm => T::Rg8Snorm,
        Rg8Uint => T::Rg8Uint,
        Rg8Sint => T::Rg8Sint,
        Rgba8Unorm => T::Rgba8Unorm,
        Rgba8Snorm => T::Rgba8Snorm,
        Rgba8Uint => T::Rgba8Uint,
        Rgba8Sint => T::Rgba8Sint,
        Rgba8Srgb => T::Rgba8UnormSrgb,
        R16Unorm => T::R16Unorm,
        R16Snorm => T::R16Snorm,
        R16Uint => T::R16Uint,
        R16Sint => T::R16Sint,
        R16Sfloat => T::R16Float,
        Rg16Unorm => T::Rg16Unorm,
        Rg16Snorm => T::Rg16Snorm,
        Rg16Uint => T::Rg16Uint,
        Rg16Sint => T::Rg16Sint,
        Rg16Sfloat => T::Rg16Float,
        Rgba16Unorm => T::Rgba16Unorm,
        Rgba16Snorm => T::Rgba16Snorm,
        Rgba16Uint => T::Rgba16Uint,
        Rgba16Sint => T::Rgba16Sint,
        Rgba16Sfloat => T::Rgba16Float,
        R32Uint => T::R32Uint,
        R32Sint => T::R32Sint,
        R32Sfloat => T::R32Float,
        Rg32Uint => T::Rg32Uint,
        Rg32Sint => T::Rg32Sint,
        Rg32Sfloat => T::Rg32Float,
        Rgba32Uint => T::Rgba32Uint,
        Rgba32Sint => T::Rgba32Sint,
        Rgba32Sfloat => T::Rgba32Float,
        R10G10B10A2Unorm => T::Rgb10a2Unorm,
        R10G10B10A2Uint => T::Rgb10a2Uint,
        R11G11B10Ufloat => T::Rg11b10Ufloat,
        R9G9B9E5Ufloat => T::Rgb9e5Ufloat,
        Rgb32Uint | Rgb32Sint | Rgb32Sfloat | MaxNum => {
            return Err(WgpuNrdError::UnsupportedNrdFormat(f));
        }
    };
    Ok(out)
}

/// Full-resolution extent from common settings and optional downsample factor (1 = full).
pub fn pool_extent(
    resource_width: u32,
    resource_height: u32,
    downsample_factor: u16,
) -> wgpu::Extent3d {
    let d = downsample_factor.max(1) as u32;
    wgpu::Extent3d {
        width: (resource_width / d).max(1),
        height: (resource_height / d).max(1),
        depth_or_array_layers: 1,
    }
}

/// Sample type for [`wgpu::BindingType::Texture`] when the bound view uses `fmt`.
///
/// Wraps [`wgpu::TextureFormat::sample_type`] with no aspect / no extra features, matching
/// validation when creating bind groups (float vs uint vs sint vs depth, and filterable float).
pub fn wgpu_texture_binding_sample_type(
    fmt: wgpu::TextureFormat,
) -> Option<wgpu::TextureSampleType> {
    fmt.sample_type(None, None)
}

/// Raw `nrd_Format` value as `Format` enum (for `ffi::nrd_TextureDesc`).
pub fn format_from_raw(raw: u32) -> Result<Format, WgpuNrdError> {
    if raw >= ffi::nrd_Format_MAX_NUM {
        return Err(WgpuNrdError::UnsupportedNrdFormat(Format::MaxNum));
    }
    Ok(unsafe { std::mem::transmute::<u32, Format>(raw) })
}

/// Default wgpu storage texture format for user (non-pool) NRD resources when SPIR-V reports
/// `ImageFormat::Unknown` (StorageImageWriteWithoutFormat). Pool textures use [`nrd_TextureDesc::format`](ffi::nrd_TextureDesc).
pub fn user_resource_storage_wgpu_format(ty: ResourceType) -> Option<wgpu::TextureFormat> {
    use ResourceType::*;
    use wgpu::TextureFormat as T;
    Some(match ty {
        InMv => T::Rg16Float,
        InNormalRoughness => T::Rgba8Unorm,
        InViewz => T::R32Float,
        InDiffConfidence | InSpecConfidence => T::R8Unorm,
        InDisocclusionThresholdMix => T::R16Float,
        InDiffRadianceHitdist
        | InSpecRadianceHitdist
        | InDiffHitdist
        | InSpecHitdist
        | InDiffDirectionHitdist => T::Rgba16Float,
        InDiffSh0 | InDiffSh1 | InSpecSh0 | InSpecSh1 => T::Rgba16Float,
        InPenumbra | InTranslucency | InSignal => T::Rgba16Float,
        OutDiffRadianceHitdist
        | OutSpecRadianceHitdist
        | OutDiffSh0
        | OutDiffSh1
        | OutSpecSh0
        | OutSpecSh1 => T::Rgba16Float,
        OutDiffHitdist | OutSpecHitdist | OutDiffDirectionHitdist => T::Rgba16Float,
        OutShadowTranslucency | OutSignal | OutValidation => T::Rgba16Float,
        TransientPool | PermanentPool | MaxNum => return None,
    })
}

/// Resolve the concrete [`wgpu::TextureFormat`] for a storage texture from NRD metadata (pool
/// descriptor or user resource type).
pub fn wgpu_format_for_resource_binding(
    rd: &nrd_sys::ResourceBinding,
    permanent: &[ffi::nrd_TextureDesc],
    transient: &[ffi::nrd_TextureDesc],
) -> Result<Option<wgpu::TextureFormat>, WgpuNrdError> {
    match rd.resource_type {
        ResourceType::TransientPool => {
            let d = transient.get(rd.index_in_pool as usize).ok_or(
                WgpuNrdError::MissingResource {
                    resource_type: ResourceType::TransientPool,
                    index_in_pool: rd.index_in_pool,
                },
            )?;
            Ok(Some(nrd_format_to_wgpu(format_from_raw(d.format)?)?))
        }
        ResourceType::PermanentPool => {
            let d = permanent.get(rd.index_in_pool as usize).ok_or(
                WgpuNrdError::MissingResource {
                    resource_type: ResourceType::PermanentPool,
                    index_in_pool: rd.index_in_pool,
                },
            )?;
            Ok(Some(nrd_format_to_wgpu(format_from_raw(d.format)?)?))
        }
        _ => Ok(user_resource_storage_wgpu_format(rd.resource_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nrd_sys::Format;

    #[test]
    fn rgba16_float_maps() {
        assert_eq!(
            nrd_format_to_wgpu(Format::Rgba16Sfloat).unwrap(),
            wgpu::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn pool_extent_downsample_2() {
        let e = pool_extent(1920, 1080, 2);
        assert_eq!(e.width, 960);
        assert_eq!(e.height, 540);
    }
}
