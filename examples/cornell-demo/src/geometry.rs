//! Cornell box triangle mesh for BLAS + per-triangle metadata for the ray trace shader.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlasVertex {
    pub pos: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TriangleMeta {
    pub normal: [f32; 4],
    pub material_id: u32,
    pub _pad: [u32; 3],
}

pub const MAT_WHITE: u32 = 0;
pub const MAT_RED: u32 = 1;
pub const MAT_GREEN: u32 = 2;
pub const MAT_EMISSIVE: u32 = 3;

fn push_tri(
    vertices: &mut Vec<BlasVertex>,
    indices: &mut Vec<u16>,
    meta: &mut Vec<TriangleMeta>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    material_id: u32,
) {
    let n = (b - a).cross(c - a).normalize();
    let base = vertices.len() as u16;
    vertices.push(BlasVertex { pos: a.to_array() });
    vertices.push(BlasVertex { pos: b.to_array() });
    vertices.push(BlasVertex { pos: c.to_array() });
    indices.extend([base, base + 1, base + 2]);
    meta.push(TriangleMeta {
        normal: [n.x, n.y, n.z, 0.0],
        material_id,
        _pad: [0; 3],
    });
}

/// Room interior: x,y,z in [0, 1]. Open at z = 0 (camera looks from -z).
pub fn build_cornell_mesh() -> (Vec<BlasVertex>, Vec<u16>, Vec<TriangleMeta>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut meta = Vec::new();

    // Left wall (red), x = 0, inward normal +X
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 1.0),
        MAT_RED,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        MAT_RED,
    );

    // Right wall (green), x = 1, inward normal -X
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0),
        MAT_GREEN,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        MAT_GREEN,
    );

    // Floor y = 0, inward +Y
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        MAT_WHITE,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        MAT_WHITE,
    );

    // Ceiling y = 1, inward -Y
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        MAT_WHITE,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        MAT_WHITE,
    );

    // Back wall z = 1, inward -Z
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        MAT_WHITE,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        MAT_WHITE,
    );

    // Area light: horizontal quad facing down (-Y), slightly below ceiling
    let lp = 0.18_f32;
    let ly = 0.995;
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.5 - lp, ly, 0.5 - lp),
        Vec3::new(0.5 + lp, ly, 0.5 - lp),
        Vec3::new(0.5 + lp, ly, 0.5 + lp),
        MAT_EMISSIVE,
    );
    push_tri(
        &mut vertices,
        &mut indices,
        &mut meta,
        Vec3::new(0.5 - lp, ly, 0.5 - lp),
        Vec3::new(0.5 + lp, ly, 0.5 + lp),
        Vec3::new(0.5 - lp, ly, 0.5 + lp),
        MAT_EMISSIVE,
    );

    // Tall white box: same footprint as classic Cornell (rotated ~-15° about Y through center).
    let tall_center = Vec3::new(0.35, 0.25, 0.45);
    let tall_half = Vec3::new(0.125, 0.25, 0.175);
    push_oriented_box(
        &mut vertices,
        &mut indices,
        &mut meta,
        tall_center,
        tall_half,
        (-15.0_f32).to_radians(),
        MAT_WHITE,
    );

    // Short white box: ~+15° about Y (classic Cornell pairing).
    let short_center = Vec3::new(0.65, 0.125, 0.25);
    let short_half = Vec3::new(0.125, 0.125, 0.125);
    push_oriented_box(
        &mut vertices,
        &mut indices,
        &mut meta,
        short_center,
        short_half,
        15.0_f32.to_radians(),
        MAT_WHITE,
    );

    (vertices, indices, meta)
}

/// `center` is the box center; `half_extents` are half-sizes along local X/Y/Z before `yaw_rad` about Y.
fn push_oriented_box(
    vertices: &mut Vec<BlasVertex>,
    indices: &mut Vec<u16>,
    meta: &mut Vec<TriangleMeta>,
    center: Vec3,
    half_extents: Vec3,
    yaw_rad: f32,
    material_id: u32,
) {
    let rot = Mat3::from_rotation_y(yaw_rad);
    let hx = half_extents.x;
    let hy = half_extents.y;
    let hz = half_extents.z;
    let local = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ];
    let c: [Vec3; 8] = std::array::from_fn(|i| center + rot * local[i]);
    // Winding: normals out of the solid (into room air). Room walls use the opposite
    // convention (normals into the air volume); the old box order matched walls and was wrong here.
    // bottom −Y
    push_tri(vertices, indices, meta, c[0], c[1], c[2], material_id);
    push_tri(vertices, indices, meta, c[0], c[2], c[3], material_id);
    // top +Y
    push_tri(vertices, indices, meta, c[4], c[6], c[5], material_id);
    push_tri(vertices, indices, meta, c[4], c[7], c[6], material_id);
    // min local Z face −Z
    push_tri(vertices, indices, meta, c[0], c[5], c[1], material_id);
    push_tri(vertices, indices, meta, c[0], c[4], c[5], material_id);
    // max local Z face +Z
    push_tri(vertices, indices, meta, c[2], c[7], c[3], material_id);
    push_tri(vertices, indices, meta, c[2], c[6], c[7], material_id);
    // min local X face −X
    push_tri(vertices, indices, meta, c[3], c[4], c[0], material_id);
    push_tri(vertices, indices, meta, c[3], c[7], c[4], material_id);
    // max local X face +X
    push_tri(vertices, indices, meta, c[1], c[6], c[2], material_id);
    push_tri(vertices, indices, meta, c[1], c[5], c[6], material_id);
}
