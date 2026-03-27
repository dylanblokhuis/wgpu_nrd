//! User-provided NRD inputs/outputs (non-pool).

use std::collections::HashMap;

use rusty_nrd::ResourceType;
use wgpu::TextureView;

/// Maps [`ResourceType`] (as `u32` discriminant) to a [`TextureView`].
///
/// Insert with `map.insert(ty as u32, view)`.
pub type UserResourceMap = HashMap<u32, TextureView>;

/// User textures for NRD dispatches.
///
/// By default, views are resolved by [`ResourceType`] via [`Self::by_type`]. If a NRD pipeline binds
/// the **same** resource type **twice** in one dispatch (e.g. sampled read + storage write), both
/// slots would otherwise resolve to the **same** view — WebGPU/wgpu forbids using one texture
/// subresource as both `TEXTURE_BINDING` and `STORAGE_BINDING` in a **single** dispatch.
///
/// Use either:
/// - [`Self::split_sampled_storage`] — pair of views `(sampled, storage)` for the same
///   [`ResourceType`] discriminant, backed by **two** distinct [`wgpu::Texture`] allocations; or
/// - [`Self::by_binding`] — explicit view per shader binding index.
#[derive(Debug, Default)]
pub struct UserResources {
    /// Resolve by [`ResourceType`] (`ty as u32`).
    pub by_type: HashMap<u32, TextureView>,
    /// Optional override by binding index in the resources bind group (set `resourcesSpaceIndex`).
    pub by_binding: HashMap<u32, TextureView>,
    /// When the same `ResourceType` is bound as both a sampled [`wgpu::BindingType::Texture`] and
    /// a [`wgpu::BindingType::StorageTexture`] in one dispatch, provide two **different** textures
    /// (two views here). The first view is used for sampled bindings, the second for storage.
    pub split_sampled_storage: HashMap<u32, (TextureView, TextureView)>,
}

impl From<HashMap<u32, TextureView>> for UserResources {
    fn from(by_type: HashMap<u32, TextureView>) -> Self {
        Self {
            by_type,
            by_binding: HashMap::new(),
            split_sampled_storage: HashMap::new(),
        }
    }
}

/// Insert a view for a resource type into [`UserResources::by_type`].
pub fn insert_user_resource(user: &mut UserResources, ty: ResourceType, view: TextureView) {
    user.by_type.insert(ty as u32, view);
}
