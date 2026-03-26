import nrd::NRD::nrd_front_end_pack_normal_and_roughness;

enable wgpu_ray_query;

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    view: mat4x4<f32>,
    frame_data: vec4<u32>,
    hit_dist_and_rough: vec4<f32>,
}

struct TriangleMeta {
    normal: vec4<f32>,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Material {
    albedo: vec4<f32>,
    emissive: vec4<f32>,
}

@group(0) @binding(0) var out_nr: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var out_mv: texture_storage_2d<rg16float, write>;
@group(0) @binding(2) var out_viewz: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var out_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var out_primary_hit_t: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var out_emissive: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;
@group(0) @binding(7) var acc_struct: acceleration_structure;
@group(0) @binding(8) var<storage, read> tri_meta: array<TriangleMeta>;
@group(0) @binding(9) var<storage, read> materials: array<Material>;

fn camera_ray(gid: vec2<u32>) -> vec3<f32> {
    let dim = textureDimensions(out_nr);
    let pixel_center = vec2<f32>(vec2<i32>(gid)) + vec2<f32>(0.5);
    let uv = pixel_center / vec2<f32>(dim);
    var d = uv * 2.0 - vec2<f32>(1.0);
    d.y = -d.y;
    return normalize((uniforms.view_inv * vec4<f32>(normalize((uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0)).xyz), 0.0)).xyz);
}

fn camera_origin() -> vec3<f32> {
    return (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(out_nr);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }

    let origin = camera_origin();
    let direction = camera_ray(gid.xy);

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, direction));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);

    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        textureStore(out_nr, gid.xy, vec4<f32>(0.5, 0.5, 0.5, 0.5));
        textureStore(out_mv, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_viewz, gid.xy, vec4<f32>(1e6, 0.0, 0.0, 0.0));
        textureStore(out_albedo, gid.xy, vec4<f32>(0.02, 0.02, 0.03, 1.0));
        textureStore(out_primary_hit_t, gid.xy, vec4<f32>(1e6, 0.0, 0.0, 0.0));
        textureStore(out_emissive, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let tm = tri_meta[hit.primitive_index];
    let n = normalize(tm.normal.xyz);
    let mat = materials[tm.material_id];
    let t = hit.t;
    let p = origin + direction * t;
    let vpos = (uniforms.view * vec4<f32>(p, 1.0)).xyz;

    let rough = 0.6;
    textureStore(out_nr, gid.xy, nrd_front_end_pack_normal_and_roughness(n, rough, 0.0));
    textureStore(out_mv, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    textureStore(out_viewz, gid.xy, vec4<f32>(vpos.z, 0.0, 0.0, 0.0));
    textureStore(out_albedo, gid.xy, vec4<f32>(mat.albedo.xyz, 1.0));
    textureStore(out_primary_hit_t, gid.xy, vec4<f32>(t, 0.0, 0.0, 0.0));
    textureStore(out_emissive, gid.xy, vec4<f32>(mat.emissive.xyz, 1.0));
}
