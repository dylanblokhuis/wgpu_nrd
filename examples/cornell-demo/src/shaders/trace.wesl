import nrd::NRD::reblur_front_end_get_norm_hit_dist;
import nrd::NRD::reblur_front_end_pack_radiance_and_norm_hit_dist;
import nrd::NRD::nrd_front_end_unpack_normal_and_roughness;

enable wgpu_ray_query;

const PI: f32 = 3.141592653589793;

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

@group(0) @binding(0) var out_diff: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var acc_struct: acceleration_structure;
@group(0) @binding(3) var<storage, read> tri_meta: array<TriangleMeta>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var in_viewz: texture_storage_2d<r32float, read>;
@group(0) @binding(6) var in_nr: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(7) var in_emissive: texture_storage_2d<rgba16float, read>;

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = (h >> 16u) ^ h;
    return h;
}

fn next_u32(state: ptr<function, u32>) -> u32 {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    return f32(next_u32(state)) * (1.0 / 4294967296.0);
}

fn orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    var b1 = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) {
        b1 = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(b1, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn sample_cosine_hemisphere_world(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(1.0 - u1, 0.0));
    let tbn = orthonormal_basis(n);
    return normalize(tbn * vec3<f32>(x, y, z));
}

fn trace_shadow_segment(origin: vec3<f32>, dir: vec3<f32>, t_max: f32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, t_max, origin, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return hit.kind != RAY_QUERY_INTERSECTION_NONE;
}

/// Monte Carlo NEE for a diffuse receiver: returns outgoing radiance / albedo (same convention as
/// the primary `direct` term in main).
fn nee_diffuse_outgoing_div_albedo(p: vec3<f32>, n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let lp = 0.18;
    let ly = 0.995;
    let light_area = 4.0 * lp * lp;
    let u_a = rand01(rng);
    let u_b = rand01(rng);
    let lx = 0.5 - lp + u_a * (2.0 * lp);
    let lz = 0.5 - lp + u_b * (2.0 * lp);
    let p_light = vec3<f32>(lx, ly, lz);
    let wi_nee = p_light - p;
    let dist_nee = length(wi_nee);
    let wi_n = wi_nee / max(dist_nee, 1e-6);
    let cos_p = max(dot(n, wi_n), 0.0);
    let cos_l = max(wi_n.y, 0.0);
    let biased = p + n * 0.0002;
    let occluded = trace_shadow_segment(biased, wi_n, dist_nee - 0.0005);
    let pdf_area = 1.0 / light_area;
    let g_nee = (cos_p * cos_l) / max(dist_nee * dist_nee, 1e-6);
    let le = materials[3].emissive.xyz;
    return select(
        (1.0 / PI) * le * g_nee * (1.0 / pdf_area),
        vec3<f32>(0.0),
        occluded,
    );
}

/// One cosine-sample path from `origin` along `dir`. `.xyz` = indirect term to accumulate
/// (outgoing at primary ÷ rho_primary); `.w` = hit distance `t`, or 0 if miss.
fn indirect_diffuse_one_bounce(
    origin: vec3<f32>,
    dir: vec3<f32>,
    rng: ptr<function, u32>,
) -> vec4<f32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let t = hit.t;
    let tm = tri_meta[hit.primitive_index];
    let mat = materials[tm.material_id];
    let n_q = normalize(tm.normal.xyz);
    let p_q = origin + dir * t;
    if (length(mat.emissive.xyz) > 0.01) {
        return vec4<f32>(mat.emissive.xyz, t);
    }
    let div_albedo_q = nee_diffuse_outgoing_div_albedo(p_q, n_q, rng);
    let rho_q = mat.albedo.xyz;
    return vec4<f32>(rho_q * div_albedo_q, t);
}

fn camera_ray(gid: vec2<u32>) -> vec3<f32> {
    let dim = textureDimensions(out_diff);
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
    let dim = textureDimensions(out_diff);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    var rng = hash_u32(gid.x + hash_u32(gid.y * 747796405u + uniforms.frame_data.x * 1597334677u));

    let origin = camera_origin();
    let direction = camera_ray(gid.xy);

    let t_prim = textureLoad(in_viewz, coord).x;
    let emissive_gbuf = textureLoad(in_emissive, coord);
    let nr_px = textureLoad(in_nr, coord);
    let unpacked_nr = nrd_front_end_unpack_normal_and_roughness(vec4<f32>(nr_px));
    let n = unpacked_nr.xyz;
    let rough = unpacked_nr.w;
    let hdp = uniforms.hit_dist_and_rough.xyz;

    if (t_prim > 1e5) {
        let sky = vec3<f32>(0.02, 0.02, 0.03);
        textureStore(
            out_diff,
            gid.xy,
            reblur_front_end_pack_radiance_and_norm_hit_dist(sky, 0.0, false),
        );
        return;
    }

    let p = origin + direction * t_prim;

    if (length(emissive_gbuf.xyz) > 0.01) {
        let norm_hit = reblur_front_end_get_norm_hit_dist(t_prim, t_prim, hdp, rough);
        textureStore(
            out_diff,
            gid.xy,
            reblur_front_end_pack_radiance_and_norm_hit_dist(emissive_gbuf.xyz, norm_hit, false),
        );
        return;
    }

    let direct = nee_diffuse_outgoing_div_albedo(p, n, &rng);

    let u1 = rand01(&rng);
    let u2 = rand01(&rng);
    let wi = sample_cosine_hemisphere_world(n, u1, u2);
    let biased_ind = p + n * 0.0002;
    let ind_sample = indirect_diffuse_one_bounce(biased_ind, wi, &rng);
    let indirect = ind_sample.xyz;
    let t_ind = ind_sample.w;

    var norm_hit_dist = reblur_front_end_get_norm_hit_dist(t_ind, t_prim, hdp, rough);

    let radiance_pre_albedo = direct + indirect;
    textureStore(
        out_diff,
        gid.xy,
        reblur_front_end_pack_radiance_and_norm_hit_dist(radiance_pre_albedo, norm_hit_dist, false),
    );
}
