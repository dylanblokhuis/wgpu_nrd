// NRD v4.17 — WGSL port of NRD.hlsli (integration helpers for denoiser I/O).
//
// IMPORTANT: Match `NRD_NORMAL_ENCODING` and `NRD_ROUGHNESS_ENCODING` to your NRD
// library build (see NRDConfig.hlsli in the NRD SDK). Defaults mirror typical
// R10G10B10A2 + sqrt-linear roughness setups.
//
// Bindings / resource macros from the HLSL header are not ported; declare
// `@group`/`@binding` resources in your pipeline shaders as required by NRD.

//=================================================================================================================================
// CONFIG — must match NRDConfig.hlsli / NRD library
//=================================================================================================================================

const NRD_NORMAL_ENCODING_RGBA8_UNORM: u32 = 0u;
const NRD_NORMAL_ENCODING_RGBA8_SNORM: u32 = 1u;
const NRD_NORMAL_ENCODING_R10G10B10A2_UNORM: u32 = 2u;
const NRD_NORMAL_ENCODING_RGBA16_UNORM: u32 = 3u;
const NRD_NORMAL_ENCODING_RGBA16_SNORM: u32 = 4u;

const NRD_ROUGHNESS_ENCODING_SQ_LINEAR: u32 = 0u;
const NRD_ROUGHNESS_ENCODING_LINEAR: u32 = 1u;
const NRD_ROUGHNESS_ENCODING_SQRT_LINEAR: u32 = 2u;

/// Set these to match your NRD build.
const NRD_NORMAL_ENCODING: u32 = NRD_NORMAL_ENCODING_R10G10B10A2_UNORM;
const NRD_ROUGHNESS_ENCODING: u32 = NRD_ROUGHNESS_ENCODING_LINEAR;

//=================================================================================================================================
// SETTINGS (defaults from NRD.hlsli)
//=================================================================================================================================

const NRD_REJITTER_VIEWZ_THRESHOLD: f32 = 0.01;
const NRD_REJITTER_AMPLITUDE: f32 = 2.0;
const NRD_MATERIAL_FACTOR_MIN_SCALE: f32 = 0.02;
const NRD_ROUGHNESS_FACTOR_MIN_SCALE: f32 = 0.1;

//=================================================================================================================================
// CONSTANTS
//=================================================================================================================================

const NRD_FP16_MAX: f32 = 65504.0;
const NRD_PI: f32 = 3.14159265358979323846;
const NRD_EPS: f32 = 1e-6;
const NRD_INF: f32 = 1e6;
const F32_MAX_FINITE: f32 = 3.402823466e38; // f32 max finite — values beyond are treated as inf

//=================================================================================================================================
// HELPERS — NaN / Inf (WGSL has no isnan/isinf builtins)
//=================================================================================================================================

fn _nrd_is_invalid_scalar(x: f32) -> bool {
    return (x != x) || (x > F32_MAX_FINITE) || (x < -F32_MAX_FINITE);
}

fn _nrd_is_invalid_vec3(x: vec3<f32>) -> bool {
    return _nrd_is_invalid_scalar(x.x) || _nrd_is_invalid_scalar(x.y) || _nrd_is_invalid_scalar(x.z);
}

//=================================================================================================================================
// MISC
//=================================================================================================================================

fn _nrd_safe_normalize(v: vec3<f32>) -> vec3<f32> {
    return v * inverseSqrt(dot(v, v) + 1e-9);
}

fn _nrd_encode_normal_roughness_101010(n: vec3<f32>, roughness: f32) -> vec3<f32> {
    var n_m = n / (abs(n.x) + abs(n.y) + abs(n.z));

    var r: vec3<f32>;
    r.y = n_m.y * 0.5 + 0.5;
    r.x = n_m.x * 0.5 + r.y;
    r.y -= n_m.x * 0.5;

    var r_rough = max(roughness, 1.5 / 512.0);
    let s = select(r_rough, -r_rough, n_m.z < 0.0);
    r.z = s * 0.5 + 0.5;

    return r;
}

fn _nrd_decode_normal_roughness_101010(p: vec3<f32>) -> vec4<f32> {
    let t = p.z * 2.0 - 1.0;

    var r: vec4<f32>;
    r.x = p.x - p.y;
    r.y = p.x + p.y - 1.0;
    r.z = select(1.0, -1.0, t < 0.0);
    r.z *= 1.0 - abs(r.x) - abs(r.y);
    r.w = abs(t);

    return r;
}

fn _nrd_luminance(linear_color: vec3<f32>) -> f32 {
    return dot(linear_color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn _nrd_linear_to_ycocg(color: vec3<f32>) -> vec3<f32> {
    let y = dot(color, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(color, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(color, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

fn _nrd_ycocg_to_linear(color: vec3<f32>) -> vec3<f32> {
    let t = color.x - color.z;
    var r: vec3<f32>;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;
    return max(r, vec3<f32>(0.0));
}

fn _nrd_ycocg_to_linear_corrected(y: f32, y0: f32, cocg: vec2<f32>) -> vec3<f32> {
    return _nrd_ycocg_to_linear(vec3<f32>(
        max(y, 0.0),
        cocg * (y + NRD_EPS) / (y0 + NRD_EPS),
    ));
}

fn _nrd_get_specular_dominant_factor(no_v: f32, roughness: f32) -> f32 {
    let a = 0.298475 * log(39.4115 - 39.0029 * roughness);
    let dominant_factor_pow = pow(clamp(1.0 - no_v, 0.0, 1.0), 10.8649);
    let dominant_factor = dominant_factor_pow * (1.0 - a) + a;
    return clamp(dominant_factor, 0.0, 1.0);
}

fn _nrd_get_specular_dominant_direction(n: vec3<f32>, v: vec3<f32>, dominant_factor: f32) -> vec3<f32> {
    let r = reflect(-v, n);
    let d = mix(n, r, dominant_factor);
    return normalize(d);
}

fn _nrd_pow5(x: f32) -> f32 {
    return pow(clamp(1.0 - x, 0.0, 1.0), 5.0);
}

fn _nrd_distribution_term(roughness: f32, no_h: f32) -> f32 {
    let m = roughness * roughness;
    let m2 = m * m;
    let t = (no_h * m2 - no_h) * no_h + 1.0;
    let a = m / t;
    let d = a * a;
    return d / NRD_PI;
}

fn _nrd_geometry_term(roughness: f32, no_l: f32, no_v: f32) -> f32 {
    let m = roughness * roughness;
    let m2 = m * m;
    let a = no_v * sqrt((no_l - m2 * no_l) * no_l + m2);
    let b = no_l * sqrt((no_v - m2 * no_v) * no_v + m2);
    return 0.5 / (a + b);
}

fn _nrd_diffuse_term(roughness: f32, no_l: f32, no_v: f32, vo_h: f32) -> f32 {
    let f = 2.0 * vo_h * vo_h * roughness - 0.5;
    let fd_v = f * _nrd_pow5(no_v) + 1.0;
    let fd_l = f * _nrd_pow5(no_l) + 1.0;
    let d = fd_v * fd_l;
    return d / NRD_PI;
}

fn _nrd_compute_brdfs(ld: vec3<f32>, ls: vec3<f32>, n: vec3<f32>, v: vec3<f32>, roughness: f32) -> vec2<f32> {
    let no_v = abs(dot(n, v));
    var result: vec2<f32>;

    {
        let h = normalize(ld + v);
        let no_l = clamp(dot(n, ld), 0.0, 1.0);
        let vo_h = abs(dot(v, h));
        let k_diff = _nrd_diffuse_term(roughness, no_l, no_v, vo_h);
        result.x = k_diff * no_l;
    }

    {
        let h = normalize(ls + v);
        let no_l = clamp(dot(n, ls), 0.0, 1.0);
        let no_h = clamp(dot(n, h), 0.0, 1.0);
        let d = _nrd_distribution_term(roughness, no_h);
        let g_mod = _nrd_geometry_term(roughness, no_l, no_v);
        let k_spec = d * g_mod;
        result.y = k_spec * no_l;
    }

    return result;
}

fn _nrd_environment_term_rtg(rf0: vec3<f32>, no_v: f32, roughness: f32) -> vec3<f32> {
    let m = clamp(roughness * roughness, 0.0, 1.0);

    let x = vec4<f32>(1.0, no_v, no_v * no_v, no_v * no_v * no_v);
    let y = vec4<f32>(1.0, m, m * m, m * m * m);

    let m1 = mat2x2<f32>(
        vec2<f32>(0.99044, 1.29678),
        vec2<f32>(-1.28514, -0.755907),
    );
    let m2 = mat3x3<f32>(
        vec3<f32>(1.0, 20.3225, 121.563),
        vec3<f32>(2.92338, -27.0302, 626.13),
        vec3<f32>(59.4188, 222.592, 316.627),
    );
    let m3 = mat2x2<f32>(
        vec2<f32>(0.0365463, 9.0632),
        vec2<f32>(3.32707, -9.04756),
    );
    let m4 = mat3x3<f32>(
        vec3<f32>(1.0, 9.04401, 5.56589),
        vec3<f32>(3.59685, -16.3174, 19.7886),
        vec3<f32>(-1.36772, 9.22949, -20.2123),
    );

    let bias = dot(m1 * x.xy, y.xy) * (1.0 / max(dot(m2 * vec3<f32>(x.x, x.y, x.w), vec3<f32>(y.x, y.y, y.w)), NRD_EPS));
    let scale = dot(m3 * x.xy, y.xy) * (1.0 / max(dot(m4 * vec3<f32>(x.x, x.z, x.w), vec3<f32>(y.x, y.y, y.w)), NRD_EPS));

    return clamp(rf0 * scale + bias, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn _nrd_get_spec_magic_curve(roughness: f32, power: f32) -> f32 {
    var f = 1.0 - exp2(-200.0 * roughness * roughness);
    f *= pow(clamp(roughness, 0.0, 1.0), power);
    return f;
}

fn _reblur_get_hit_distance_normalization(view_z: f32, hit_dist_params: vec3<f32>, roughness: f32) -> f32 {
    let smc = _nrd_get_spec_magic_curve(roughness, 0.5);
    return (hit_dist_params.x + abs(view_z) * hit_dist_params.y) * mix(hit_dist_params.z, 1.0, smc);
}

//=================================================================================================================================
// SPHERICAL GAUSSIAN
//=================================================================================================================================

struct NrdSg {
    c0: f32,
    chroma: vec2<f32>,
    norm_hit_dist: f32,
    c1: vec3<f32>,
    sharpness: f32,
}

fn _nrd_sg_create(radiance: vec3<f32>, direction: vec3<f32>, norm_hit_dist: f32) -> NrdSg {
    let ycocg = _nrd_linear_to_ycocg(radiance);
    var sg: NrdSg;
    sg.c0 = ycocg.x;
    sg.chroma = ycocg.yz;
    sg.c1 = direction * ycocg.x;
    sg.norm_hit_dist = norm_hit_dist;
    sg.sharpness = 0.0;
    return sg;
}

fn _nrd_sg_impl_extract_direction(sg: NrdSg) -> vec3<f32> {
    return sg.c1 / max(length(sg.c1), NRD_EPS);
}

fn _nrd_sg_integral_approx(sg: NrdSg) -> f32 {
    return 2.0 * NRD_PI * (sg.c0 / sg.sharpness);
}

fn _nrd_sg_integral(sg: NrdSg) -> f32 {
    let exp_term = 1.0 - exp(-2.0 * sg.sharpness);
    return _nrd_sg_integral_approx(sg) * exp_term;
}

fn _nrd_sg_inner_product(a: NrdSg, b: NrdSg) -> f32 {
    let dir = a.sharpness * a.c1 + b.sharpness * b.c1;
    let d = length(dir);
    var c = exp(d - a.sharpness - b.sharpness);
    c *= 1.0 - exp(-2.0 * d);
    c /= max(d, NRD_EPS);
    return 2.0 * NRD_PI * c * a.c0 * b.c0;
}

//=================================================================================================================================
// FRONT-END — GENERAL
//=================================================================================================================================

fn nrd_front_end_unpack_roughness(r: vec4<f32>) -> vec4<f32> {
    var rr = r;
    if (NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM) {
        rr = abs(rr * 2.0 - 1.0);
    }
    if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR) {
        rr.w = clamp(rr.w * rr.w, 0.0, 1.0);
    } else if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR) {
        rr.w = sqrt(clamp(rr.w, 0.0, 1.0));
    }
    return rr;
}

struct NrdUnpackNrMaterial {
    nr: vec4<f32>,
    material_id: f32,
}

fn nrd_front_end_unpack_normal_and_roughness_with_material(p: vec4<f32>) -> NrdUnpackNrMaterial {
    var r: vec4<f32>;
    var material_id: f32;

    if (NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM) {
        r = _nrd_decode_normal_roughness_101010(p.xyz);
        material_id = p.w * 3.0;
    } else {
        var p_xyz = p.xyz;
        if (NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM) {
            p_xyz = p_xyz * 2.0 - 1.0;
        }
        r = vec4<f32>(p_xyz, p.w);
        material_id = 0.0;
    }

    r = vec4<f32>(_nrd_safe_normalize(r.xyz), r.w);

    if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR) {
        r.w = clamp(r.w * r.w, 0.0, 1.0);
    } else if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR) {
        r.w = sqrt(clamp(r.w, 0.0, 1.0));
    }

    return NrdUnpackNrMaterial(r, material_id);
}

fn nrd_front_end_unpack_normal_and_roughness(p: vec4<f32>) -> vec4<f32> {
    return nrd_front_end_unpack_normal_and_roughness_with_material(p).nr;
}

fn nrd_front_end_pack_normal_and_roughness(n: vec3<f32>, roughness: f32, material_id: f32) -> vec4<f32> {
    var rr = roughness;
    if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR) {
        rr = sqrt(clamp(roughness, 0.0, 1.0));
    } else if (NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR) {
        rr *= roughness;
    }

    var p: vec4<f32>;
    if (NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM) {
        p = vec4<f32>(_nrd_encode_normal_roughness_101010(n, rr), clamp(material_id / 3.0, 0.0, 1.0));
    } else {
        var nn = n / max(abs(n.x), max(abs(n.y), abs(n.z)));
        if (NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM) {
            nn = nn * 0.5 + 0.5;
        }
        p = vec4<f32>(nn, rr);
    }
    return p;
}

struct NrdMaterialFactors {
    diff_factor: vec3<f32>,
    spec_factor: vec3<f32>,
}

fn nrd_material_factors(
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    rf0: vec3<f32>,
    roughness: f32,
) -> NrdMaterialFactors {
    let no_v = abs(dot(n, v));
    let f_env = _nrd_environment_term_rtg(rf0, no_v, roughness);

    var diff_factor = (vec3<f32>(1.0) - f_env) * albedo;
    diff_factor = mix(vec3<f32>(NRD_MATERIAL_FACTOR_MIN_SCALE), vec3<f32>(1.0), diff_factor);

    var spec_factor = f_env;
    spec_factor *= mix(vec3<f32>(NRD_ROUGHNESS_FACTOR_MIN_SCALE), vec3<f32>(1.0), vec3<f32>(roughness));
    spec_factor = mix(vec3<f32>(NRD_MATERIAL_FACTOR_MIN_SCALE), vec3<f32>(1.0), spec_factor);

    return NrdMaterialFactors(diff_factor, spec_factor);
}

fn nrd_front_end_spec_hit_dist_averaging_begin() -> f32 {
    return NRD_INF;
}

fn nrd_front_end_trim_hit_distance(hit_dist: f32, threshold: f32) -> f32 {
    return select(hit_dist, 0.0, hit_dist < threshold);
}

fn nrd_front_end_spec_hit_dist_averaging_add(accumulated_spec_hit_dist: ptr<function, f32>, hit_dist: f32) {
    *accumulated_spec_hit_dist = min(
        *accumulated_spec_hit_dist,
        select(hit_dist, NRD_INF, hit_dist == 0.0),
    );
}

fn nrd_front_end_spec_hit_dist_averaging_end(accumulated_spec_hit_dist: ptr<function, f32>) {
    *accumulated_spec_hit_dist = select(*accumulated_spec_hit_dist, 0.0, *accumulated_spec_hit_dist == NRD_INF);
}

//=================================================================================================================================
// REBLUR
//=================================================================================================================================

fn reblur_front_end_get_norm_hit_dist(hit_dist: f32, view_z: f32, hit_dist_params: vec3<f32>, roughness: f32) -> f32 {
    let f = _reblur_get_hit_distance_normalization(view_z, hit_dist_params, roughness);
    return clamp(hit_dist / f, 0.0, 1.0);
}

fn reblur_front_end_pack_radiance_and_norm_hit_dist(radiance: vec3<f32>, norm_hit_dist: f32, sanitize: bool) -> vec4<f32> {
    var rad = radiance;
    var nhd = norm_hit_dist;
    if (sanitize) {
        rad = select(rad, vec3<f32>(0.0), _nrd_is_invalid_vec3(rad));
        rad = clamp(rad, vec3<f32>(0.0), vec3<f32>(NRD_FP16_MAX));
        nhd = select(nhd, 0.0, _nrd_is_invalid_scalar(nhd));
        nhd = clamp(nhd, 0.0, 1.0);
    }
    let ycocg = _nrd_linear_to_ycocg(rad);
    return vec4<f32>(ycocg, nhd);
}

struct ReblurPackShResult {
    out0: vec4<f32>,
    out1: vec4<f32>,
}

fn reblur_front_end_pack_sh(
    radiance: vec3<f32>,
    norm_hit_dist: f32,
    direction: vec3<f32>,
    sanitize: bool,
) -> ReblurPackShResult {
    var rad = radiance;
    var nhd = norm_hit_dist;
    var dir = direction;
    if (sanitize) {
        rad = select(rad, vec3<f32>(0.0), _nrd_is_invalid_vec3(rad));
        rad = clamp(rad, vec3<f32>(0.0), vec3<f32>(NRD_FP16_MAX));
        nhd = select(nhd, 0.0, _nrd_is_invalid_scalar(nhd));
        nhd = clamp(nhd, 0.0, 1.0);
        dir = select(dir, vec3<f32>(0.0), _nrd_is_invalid_vec3(dir));
        dir = clamp(dir, vec3<f32>(-1.0), vec3<f32>(1.0));
    }
    let sg = _nrd_sg_create(rad, dir, nhd);
    let out0 = vec4<f32>(sg.c0, sg.chroma, sg.norm_hit_dist);
    let out1 = vec4<f32>(sg.c1, sg.sharpness);
    return ReblurPackShResult(out0, out1);
}

fn reblur_front_end_pack_directional_occlusion(direction: vec3<f32>, norm_hit_dist: f32, sanitize: bool) -> vec4<f32> {
    var dir = direction;
    var nhd = norm_hit_dist;
    if (sanitize) {
        dir = select(dir, vec3<f32>(0.0), _nrd_is_invalid_vec3(dir));
        dir = clamp(dir, vec3<f32>(-1.0), vec3<f32>(1.0));
        nhd = select(nhd, 0.0, _nrd_is_invalid_scalar(nhd));
        nhd = clamp(nhd, 0.0, 1.0);
    }
    let sg = _nrd_sg_create(vec3<f32>(nhd), dir, nhd);
    return vec4<f32>(sg.c1, sg.c0);
}

fn reblur_back_end_unpack_radiance_and_norm_hit_dist(data: vec4<f32>) -> vec4<f32> {
    var d = data;
    d = vec4<f32>(_nrd_ycocg_to_linear(d.xyz), d.w);
    return d;
}

fn reblur_back_end_unpack_sh(sh0: vec4<f32>, sh1: vec4<f32>) -> NrdSg {
    var sg: NrdSg;
    sg.c0 = sh0.x;
    sg.chroma = sh0.yz;
    sg.norm_hit_dist = sh0.w;
    sg.c1 = sh1.xyz;
    sg.sharpness = sh1.w;
    return sg;
}

fn reblur_back_end_unpack_directional_occlusion(data: vec4<f32>) -> NrdSg {
    var sg: NrdSg;
    sg.c0 = data.w;
    sg.chroma = vec2<f32>(0.0);
    sg.norm_hit_dist = data.w;
    sg.c1 = data.xyz;
    sg.sharpness = 0.0;
    return sg;
}

//=================================================================================================================================
// RELAX
//=================================================================================================================================

fn relax_front_end_pack_radiance_and_hit_dist(radiance: vec3<f32>, hit_dist: f32, sanitize: bool) -> vec4<f32> {
    var rad = radiance;
    var hd = hit_dist;
    if (sanitize) {
        rad = select(rad, vec3<f32>(0.0), _nrd_is_invalid_vec3(rad));
        rad = clamp(rad, vec3<f32>(0.0), vec3<f32>(NRD_FP16_MAX));
        hd = select(hd, 0.0, _nrd_is_invalid_scalar(hd));
        hd = clamp(hd, 0.0, NRD_FP16_MAX);
    }
    return vec4<f32>(rad, hd);
}

fn relax_front_end_pack_sh(
    radiance: vec3<f32>,
    hit_dist: f32,
    direction: vec3<f32>,
    sanitize: bool,
) -> ReblurPackShResult {
    var rad = radiance;
    var hd = hit_dist;
    var dir = direction;
    if (sanitize) {
        rad = select(rad, vec3<f32>(0.0), _nrd_is_invalid_vec3(rad));
        rad = clamp(rad, vec3<f32>(0.0), vec3<f32>(NRD_FP16_MAX));
        hd = select(hd, 0.0, _nrd_is_invalid_scalar(hd));
        hd = clamp(hd, 0.0, NRD_FP16_MAX);
        dir = select(dir, vec3<f32>(0.0), _nrd_is_invalid_vec3(dir));
        dir = clamp(dir, vec3<f32>(-1.0), vec3<f32>(1.0));
    }
    let out0 = vec4<f32>(rad, hd);
    let out1 = vec4<f32>(dir * _nrd_luminance(rad), 0.0);
    return ReblurPackShResult(out0, out1);
}

fn relax_back_end_unpack_radiance(color: vec4<f32>) -> vec4<f32> {
    return color;
}

fn relax_back_end_unpack_sh(sh0: vec4<f32>, sh1: vec4<f32>) -> NrdSg {
    var sg: NrdSg;
    sg.c0 = sh0.x;
    sg.chroma = sh0.yz;
    sg.norm_hit_dist = sh0.w;
    sg.c1 = sh1.xyz;
    sg.sharpness = sh1.w;
    return sg;
}

//=================================================================================================================================
// SIGMA
//=================================================================================================================================

fn sigma_front_end_pack_penumbra_directional(distance_to_occluder: f32, tan_of_light_angular_radius: f32) -> f32 {
    let penumbra_size = distance_to_occluder * tan_of_light_angular_radius;
    let penumbra_radius = penumbra_size * 0.5;
    return select(
        min(penumbra_radius, 32768.0),
        NRD_FP16_MAX,
        distance_to_occluder >= NRD_FP16_MAX,
    );
}

fn sigma_front_end_pack_penumbra_local(
    distance_to_occluder: f32,
    distance_to_light: f32,
    light_size: f32,
) -> f32 {
    let penumbra_size = light_size * distance_to_occluder / max(distance_to_light - distance_to_occluder, NRD_EPS);
    let penumbra_radius = penumbra_size * 0.5;
    return select(
        min(penumbra_radius, 32768.0),
        NRD_FP16_MAX,
        distance_to_occluder >= NRD_FP16_MAX,
    );
}

fn sigma_front_end_pack_translucency(distance_to_occluder: f32, translucency: vec3<f32>) -> vec4<f32> {
    let t = clamp(translucency, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(f32(distance_to_occluder >= NRD_FP16_MAX), t);
}

fn sigma_back_end_unpack_shadow(shadow: vec4<f32>) -> vec4<f32> {
    return shadow * shadow;
}

//=================================================================================================================================
// HIGH QUALITY RESOLVE (SPHERICAL GAUSSIAN)
//=================================================================================================================================

fn nrd_sg_extract_color(sg: NrdSg) -> vec3<f32> {
    return _nrd_ycocg_to_linear(vec3<f32>(sg.c0, sg.chroma));
}

fn nrd_sg_extract_direction(sg: NrdSg) -> vec3<f32> {
    return _nrd_sg_impl_extract_direction(sg);
}

fn nrd_sg_extract_roughness_aa(sg: NrdSg) -> f32 {
    return sg.sharpness;
}

fn nrd_sg_rotate(sg: ptr<function, NrdSg>, rotation: mat3x3<f32>) {
    (*sg).c1 = rotation * (*sg).c1;
}

fn nrd_sg_resolve_diffuse(sg: NrdSg, n: vec3<f32>, v: vec3<f32>, roughness: f32) -> vec3<f32> {
    let l = _nrd_sg_impl_extract_direction(sg);
    let no_l = clamp(dot(n, l), 0.0, 1.0);

    var light: NrdSg = NrdSg(0.0, vec2<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
    light.sharpness = 2.0;
    light.c0 = sg.c0 * light.sharpness;
    light.c1 = l;

    var ndf: NrdSg = NrdSg(0.0, vec2<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
    ndf.c0 = 1.0;
    ndf.c1 = n;
    ndf.sharpness = 2.0;

    var y = _nrd_sg_inner_product(ndf, light);

    let h = normalize(l + v);
    let no_v = abs(dot(n, v));
    let vo_h = abs(dot(v, h));
    let k_diff = _nrd_diffuse_term(roughness, no_l, no_v, vo_h);
    y *= k_diff;
    y *= mix(1.0, mix(1.5, 0.6, roughness), _nrd_pow5(no_v));

    y = max(y, sg.c0 / NRD_PI);

    return _nrd_ycocg_to_linear_corrected(y, sg.c0, sg.chroma);
}

fn nrd_sg_resolve_specular(sg: NrdSg, n: vec3<f32>, v: vec3<f32>, roughness: f32) -> vec3<f32> {
    let rr = max(roughness, 0.03);
    let m = rr * rr;
    let m2 = m * m;

    let l = _nrd_sg_impl_extract_direction(sg);
    let no_l = clamp(dot(n, l), 0.0, 1.0);

    let h = normalize(l + v);
    var no_v = abs(dot(n, v));
    let vo_h = abs(dot(v, h));

    no_v = mix(0.02, 1.0, no_v);

    var light: NrdSg = NrdSg(0.0, vec2<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
    light.sharpness = 2.0 / m2;
    light.c0 = sg.c0 * light.sharpness;
    light.c1 = l;

    let ndf_sharpness = 0.5 / max(m2 * vo_h, 1e-8);

    var warped_ndf: NrdSg = NrdSg(0.0, vec2<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
    warped_ndf.c0 = 1.0;
    warped_ndf.c1 = l;
    warped_ndf.sharpness = ndf_sharpness;

    var y = _nrd_sg_inner_product(warped_ndf, light);

    let g_mod = _nrd_geometry_term(rr, no_l, no_v);
    y *= g_mod * no_l;

    y *= mix(mix(0.1, 0.4, m2), 0.8, no_v);

    y = max(y, sg.c0 / NRD_PI);

    return _nrd_ycocg_to_linear_corrected(y, sg.c0, sg.chroma);
}

fn nrd_sg_rejitter(
    diff_sg: NrdSg,
    spec_sg: NrdSg,
    v: vec3<f32>,
    roughness: f32,
    z: f32,
    ze: f32,
    zw: f32,
    zn: f32,
    zs: f32,
    n: vec3<f32>,
    ne: vec3<f32>,
    nw: vec3<f32>,
    nn: vec3<f32>,
    ns: vec3<f32>,
) -> vec2<f32> {
    var ld = _nrd_sg_impl_extract_direction(diff_sg);
    var ls = _nrd_sg_impl_extract_direction(spec_sg);
    ls = normalize(mix(v, ls, roughness));

    let brdf_center = _nrd_compute_brdfs(ld, ls, n, v, roughness);
    var brdf_average = _nrd_compute_brdfs(ld, ls, ne, v, roughness);
    brdf_average += _nrd_compute_brdfs(ld, ls, nn, v, roughness);
    brdf_average += _nrd_compute_brdfs(ld, ls, nw, v, roughness);
    brdf_average += _nrd_compute_brdfs(ld, ls, ns, v, roughness);
    brdf_average *= 0.25;

    var j = (brdf_center + NRD_EPS) / (brdf_average + NRD_EPS);
    j = clamp(j, vec2<f32>(1.0 / NRD_REJITTER_AMPLITUDE), vec2<f32>(NRD_REJITTER_AMPLITUDE));

    let no_v = abs(dot(n, v));
    let z_threshold = NRD_REJITTER_VIEWZ_THRESHOLD * abs(z) / (no_v * 0.95 + 0.05);
    let w = step(
        vec4<f32>(z_threshold),
        abs(vec4<f32>(ze, zw, zn, zs) - vec4<f32>(z)),
    );
    let is_symmetrical = dot(w, vec4<f32>(1.0)) > 3.5;

    return select(vec2<f32>(1.0), j, is_symmetrical);
}

//=================================================================================================================================
// MEDIUM QUALITY RESOLVE (SPHERICAL HARMONICS)
//=================================================================================================================================

fn nrd_sh_resolve_diffuse(sh: NrdSg, n: vec3<f32>) -> vec3<f32> {
    let k0 = 1.0 / NRD_PI;
    let k1 = 3.0 / NRD_PI;
    let y = sh.c0 * k0 + dot(sh.c1, n) * k1;
    return _nrd_ycocg_to_linear_corrected(y, sh.c0, sh.chroma);
}

fn nrd_sh_resolve_specular(sh: NrdSg, n: vec3<f32>, v: vec3<f32>, roughness: f32) -> vec3<f32> {
    let k0 = 1.0 / NRD_PI;
    let k1 = 3.0 / NRD_PI;
    let no_v = abs(dot(n, v));
    let f = _nrd_get_specular_dominant_factor(no_v, roughness);
    let d = _nrd_get_specular_dominant_direction(n, v, f);
    let y = sh.c0 * k0 + dot(sh.c1, d) * k1;
    return _nrd_ycocg_to_linear_corrected(y, sh.c0, sh.chroma);
}

//=================================================================================================================================
// MISC
//=================================================================================================================================

fn nrd_is_valid_radiance(radiance: vec3<f32>) -> bool {
    return !_nrd_is_invalid_vec3(radiance);
}

fn reblur_get_hit_dist(norm_hit_dist: f32, view_z: f32, hit_dist_params: vec3<f32>, roughness: f32) -> f32 {
    let scale = _reblur_get_hit_distance_normalization(view_z, hit_dist_params, roughness);
    return norm_hit_dist * scale;
}

fn nrd_get_normalized_strand_thickness(strand_thickness: f32, pixel_size: f32) -> f32 {
    return clamp(0.5 * pixel_size / (strand_thickness + NRD_EPS), 0.0, 1.0);
}
