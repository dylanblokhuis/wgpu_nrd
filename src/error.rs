//! Error types for `wgpu_nrd`.

/// Errors produced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum WgpuNrdError {
    /// NRD C API error.
    #[error("NRD: {0}")]
    Nrd(#[from] nrd_sys::Error),
    /// wgpu validation or device error (string from `wgpu::Error`).
    #[error("wgpu: {0}")]
    Wgpu(String),
    /// Unsupported `nrd_sys::Format` for `wgpu::TextureFormat`.
    #[error("unsupported NRD texture format: {0:?}")]
    UnsupportedNrdFormat(nrd_sys::Format),
    /// Pool or user resource binding could not be resolved for a dispatch.
    #[error("missing texture view for resource {resource_type:?} (descriptor index {index_in_pool})")]
    MissingResource {
        resource_type: nrd_sys::ResourceType,
        index_in_pool: u16,
    },
    /// SPIR-V bytecode length is not a multiple of 4.
    #[error("invalid SPIR-V size {0}")]
    InvalidSpirvSize(usize),
    /// Failed to parse SPIR-V for reflection (workgroup size).
    #[error("SPIR-V parse: {0}")]
    SpirvReflect(String),
    /// Shader entry point not found in SPIR-V module.
    #[error("entry point {0:?} not found in SPIR-V")]
    EntryPointNotFound(String),
    /// `embed-msl` feature is required on this backend but was not enabled.
    #[error("Metal backend requires wgpu_nrd feature `embed-msl` (and nrd-sys built with embedded MSL)")]
    EmbedMslRequired,
    /// Pool texture needs texture and/or storage binding, but the format allows none of those usages on this adapter.
    #[error("texture format {format:?} cannot be used for NRD pools on this adapter (allowed usages: {allowed:?})")]
    PoolTextureFormatNotBindable {
        format: wgpu::TextureFormat,
        allowed: wgpu::TextureUsages,
    },
}
