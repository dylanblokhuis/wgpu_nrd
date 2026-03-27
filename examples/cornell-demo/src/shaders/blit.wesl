import nrd::NRD::reblur_back_end_unpack_radiance_and_norm_hit_dist;
import nrd::NRD::nrd_front_end_unpack_normal_and_roughness;

struct BlitUniforms {
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(f32(x) * 2.0, f32(y) * 2.0);
    var out: VertexOutput;
    out.position = vec4<f32>(tc.x * 2.0 - 1.0, 1.0 - tc.y * 2.0, 0.0, 1.0);
    out.tex_coords = tc;
    return out;
}

@group(0) @binding(0) var t_radiance_denoised: texture_2d<f32>;
@group(0) @binding(1) var t_radiance_raw: texture_2d<f32>;
@group(0) @binding(2) var t_albedo: texture_2d<f32>;
@group(0) @binding(3) var t_nr: texture_2d<f32>;
@group(0) @binding(4) var t_view_z: texture_2d<f32>;
@group(0) @binding(5) var r_sampler: sampler;
@group(0) @binding(6) var<uniform> blit_u: BlitUniforms;
@group(0) @binding(7) var t_nrd_validation: texture_2d<f32>;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vertex.tex_coords;
    let mode = blit_u.mode;

    // 4 = NRD validation layer (4×4 grid, see NRD README "VALIDATION LAYER").
    if (mode == 4u) {
        let v = textureSample(t_nrd_validation, r_sampler, uv);
        return vec4<f32>(v.rgb, 1.0);
    }

    // 2 = normals (world normals, packed NRD gbuffer).
    if (mode == 2u) {
        let nr = textureSample(t_nr, r_sampler, uv);
        let n = nrd_front_end_unpack_normal_and_roughness(nr);
        return vec4<f32>(n.xyz * 0.5 + 0.5, 1.0);
    }

    // 3 = primary ray hit distance (grayscale; sky = large sentinel).
    if (mode == 3u) {
        let dims = textureDimensions(t_view_z);
        let max_c = vec2<i32>(dims) - vec2<i32>(1, 1);
        let pix = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0, 0), max_c);
        let hit_t = textureLoad(t_view_z, pix, 0).x;
        if (hit_t > 1e5) {
            return vec4<f32>(0.02, 0.02, 0.03, 1.0);
        }
        let g = saturate(hit_t * 0.25);
        return vec4<f32>(g, g, g, 1.0);
    }

    // 0 = denoised radiance, 1 = raw trace (no denoiser).
    let rad_packed = select(
        textureSample(t_radiance_raw, r_sampler, uv),
        textureSample(t_radiance_denoised, r_sampler, uv),
        mode == 0u,
    );
    let albedo = textureSample(t_albedo, r_sampler, uv);

    let rad_unpacked = reblur_back_end_unpack_radiance_and_norm_hit_dist(rad_packed);

    // Diffuse buffer stores lighting before albedo; emissive is packed as-is (gbuffer albedo ~0 on lights).
    let has_diffuse_albedo = dot(albedo.xyz, albedo.xyz) > 1e-8;
    let lit = select(
        rad_unpacked.xyz,
        albedo.xyz * rad_unpacked.xyz,
        has_diffuse_albedo,
    );

    return vec4<f32>(lit, 1.0);
}
