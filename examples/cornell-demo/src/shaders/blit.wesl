import nrd::NRD::reblur_back_end_unpack_radiance_and_norm_hit_dist;

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

@group(0) @binding(0) var t_radiance: texture_2d<f32>;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var r_sampler: sampler;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vertex.tex_coords;
    let rad_packed = textureSample(t_radiance, r_sampler, uv);
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
