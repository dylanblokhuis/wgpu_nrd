//! Cornell box G-buffer + full-res diffuse trace (primary NEE + one cosine bounce with secondary
//! NEE for GI) + NRD ReblurDiffuse temporal/spatial denoise + blit (wgpu ray query).

mod geometry;

use std::{iter, mem};

use bytemuck::{Pod, Zeroable};
use geometry::{BlasVertex, build_cornell_mesh};
use glam::{Affine3A, Mat4, Vec3};
use sdl3::mouse::MouseButton;
use wgpu::SurfaceTargetUnsafe;
use wgpu::util::DeviceExt;
use wgpu_nrd::{
    Denoiser, DenoiserSlot, Identifier, ResourceType, UserResources, WgpuNrd,
    default_common_settings, default_reblur_settings,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtUniforms {
    view_inv: [f32; 16],
    proj_inv: [f32; 16],
    view: [f32; 16],
    /// .x = frame_index for shader RNG; rest unused.
    frame_data: [u32; 4],
    /// .xyz = NRD `ReblurHitDistanceParameters` (A, B, C); .w = surface roughness for hit-dist norm.
    hit_dist_and_rough: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMaterial {
    albedo: [f32; 4],
    emissive: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlitUniforms {
    mode: u32,
    _pad: [u32; 3],
}

#[inline]
fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
    let row_0 = mat.matrix3.row(0);
    let row_1 = mat.matrix3.row(1);
    let row_2 = mat.matrix3.row(2);
    let translation = mat.translation;
    [
        row_0.x,
        row_0.y,
        row_0.z,
        translation.x,
        row_1.x,
        row_1.y,
        row_1.z,
        translation.y,
        row_2.x,
        row_2.y,
        row_2.z,
        translation.z,
    ]
}

fn main() {
    let mut wesl = wesl::Wesl::new("src/shaders");
    wesl.add_package(&wgpu_nrd_shader::PACKAGE);

    let ctx = sdl3::init().unwrap();
    let window = ctx
        .video()
        .unwrap()
        .window("Cornell Box Denoised", 1024, 768)
        .build()
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        required_limits: wgpu::Limits {
            max_storage_textures_per_shader_stage: 16,
            ..wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
        },
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .unwrap();

    let surface = unsafe {
        instance
            .create_surface_unsafe(
                SurfaceTargetUnsafe::from_display_and_window(&window, &window).unwrap(),
            )
            .unwrap()
    };
    let cap = surface.get_capabilities(&adapter);
    let (width, height) = window.size();
    let content_scale = window.display_scale();
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: cap.formats[0],
        view_formats: vec![cap.formats[0].add_srgb_suffix()],
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        width: (width as f32 * content_scale) as u32,
        height: (height as f32 * content_scale) as u32,
        desired_maximum_frame_latency: 3,
        present_mode: wgpu::PresentMode::AutoVsync,
    };
    surface.configure(&device, &surface_config);

    let denoisers = [DenoiserSlot {
        identifier: Identifier(0),
        denoiser: Denoiser::ReblurDiffuse,
    }];

    let mut common_for_pipelines = default_common_settings();
    common_for_pipelines.resourceSize = [surface_config.width as u16, surface_config.height as u16];
    common_for_pipelines.resourceSizePrev =
        [surface_config.width as u16, surface_config.height as u16];
    common_for_pipelines.rectSize = [surface_config.width as u16, surface_config.height as u16];
    common_for_pipelines.rectSizePrev = [surface_config.width as u16, surface_config.height as u16];
    common_for_pipelines.viewZScale = 1.0;
    // Needed so `clone_dispatch_resource_lists` sees the validation dispatch and pool layouts match.
    common_for_pipelines.enableValidation = true;

    let mut reblur_for_pipelines = default_reblur_settings();
    reblur_for_pipelines.hitDistanceReconstructionMode = 1;

    let mut wgpu_nrd = WgpuNrd::new(
        &device,
        &adapter,
        &denoisers,
        adapter.get_info().backend,
        surface_config.width,
        surface_config.height,
        &[denoisers[0].identifier],
        |inst| {
            inst.set_common_settings(&common_for_pipelines)?;
            inst.set_reblur_settings(denoisers[0].identifier, &reblur_for_pipelines)?;
            Ok(())
        },
    )
    .unwrap();

    let (vertex_data, index_data, tri_meta) = build_cornell_mesh();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cornell_vertices"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });

    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cornell_indices"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let tri_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cornell_tri_meta"),
        contents: bytemuck::cast_slice(&tri_meta),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gpu_materials: [GpuMaterial; 4] = [
        GpuMaterial {
            albedo: [0.73, 0.73, 0.73, 1.0],
            emissive: [0.0; 4],
        },
        GpuMaterial {
            albedo: [0.63, 0.065, 0.05, 1.0],
            emissive: [0.0; 4],
        },
        GpuMaterial {
            albedo: [0.14, 0.45, 0.091, 1.0],
            emissive: [0.0; 4],
        },
        GpuMaterial {
            albedo: [0.0; 4],
            emissive: [15.0, 15.0, 15.0, 1.0],
        },
    ];
    let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cornell_materials"),
        contents: bytemuck::cast_slice(&gpu_materials),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint16),
        index_count: Some(index_data.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("cornell_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("cornell_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        affine_to_rows(&Affine3A::IDENTITY),
        0,
        0xff,
    ));

    let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gbuffer"),
        source: wgpu::ShaderSource::Wgsl(
            wesl.compile(&"package::gbuffer".parse().unwrap())
                .unwrap()
                .to_string()
                .into(),
        ),
    });
    let diffuse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trace_diffuse"),
        source: wgpu::ShaderSource::Wgsl(
            wesl.compile(&"package::trace".parse().unwrap())
                .unwrap()
                .to_string()
                .into(),
        ),
    });
    let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(
            wesl.compile(&"package::blit".parse().unwrap())
                .unwrap()
                .to_string()
                .into(),
        ),
    });

    let gbuffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cornell_gbuffer_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rg16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let diffuse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cornell_diffuse_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let gbuffer_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cornell_gbuffer_pl"),
        bind_group_layouts: &[Some(&gbuffer_bgl)],
        immediate_size: 0,
    });
    let diffuse_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cornell_diffuse_pl"),
        bind_group_layouts: &[Some(&diffuse_bgl)],
        immediate_size: 0,
    });

    let gbuffer_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cornell_gbuffer"),
        layout: Some(&gbuffer_pl),
        module: &gbuffer_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cornell_diffuse"),
        layout: Some(&diffuse_pl),
        module: &diffuse_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("cornell_blit_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let blit_bgl_owned = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cornell_blit_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cornell_blit_pl"),
        bind_group_layouts: &[Some(&blit_bgl_owned)],
        immediate_size: 0,
    });

    let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cornell_blit"),
        layout: Some(&blit_pl),
        vertex: wgpu::VertexState {
            module: &blit_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &blit_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("as_build"),
    });
    encoder.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: mem::size_of::<BlasVertex>() as u64,
                index_buffer: Some(&index_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas),
    );
    queue.submit(Some(encoder.finish()));

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cornell_rt_uniforms"),
        size: mem::size_of::<RtUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let blit_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cornell_blit_uniforms"),
        size: mem::size_of::<BlitUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut frame_index: u32 = 0;

    let gbuffer_extent = wgpu::Extent3d {
        width: surface_config.width,
        height: surface_config.height,
        depth_or_array_layers: 1,
    };

    let in_mv = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("in_mv"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let in_view_z = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        label: Some("in_view_z"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });
    let in_view_z_storage = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        label: Some("in_view_z_storage"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let in_normal_roughness = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("in_normal_roughness"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let in_albedo = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("in_albedo"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let in_emissive = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("in_emissive"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let in_diff_radiance_hitdist = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("in_diff_radiance_hitdist"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });
    let out_diff_radiance_hitdist = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("out_diff_radiance_hitdist"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });
    let out_validation = device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
        label: Some("out_validation"),
        size: gbuffer_extent,
        mip_level_count: 1,
    });

    let v_in_mv = in_mv.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_view_z = in_view_z.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_view_z_storage =
        in_view_z_storage.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_nr = in_normal_roughness.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_albedo = in_albedo.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_emissive = in_emissive.create_view(&wgpu::TextureViewDescriptor::default());
    let v_in_diff = in_diff_radiance_hitdist.create_view(&wgpu::TextureViewDescriptor::default());
    let v_out_diff = out_diff_radiance_hitdist.create_view(&wgpu::TextureViewDescriptor::default());
    let v_out_validation = out_validation.create_view(&wgpu::TextureViewDescriptor::default());

    let gbuffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cornell_gbuffer_bg"),
        layout: &gbuffer_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&v_in_nr),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&v_in_mv),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&v_in_view_z),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&v_in_albedo),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&v_in_emissive),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: tri_meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: materials_buf.as_entire_binding(),
            },
        ],
    });

    let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cornell_diffuse_bg"),
        layout: &diffuse_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&v_in_diff),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tri_meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: materials_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&v_in_view_z),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&v_in_nr),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&v_in_emissive),
            },
        ],
    });

    // Blit `mode` (see `blit.wesl`): 0 denoised, 1 raw, 2 normals, 3 depth, 4 NRD validation grid.
    let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cornell_blit_bg"),
        layout: &blit_bgl_owned,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&v_out_diff),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&v_in_diff),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&v_in_albedo),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&v_in_nr),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&v_in_view_z),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&blit_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: blit_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&v_out_validation),
            },
        ],
    });

    // Left-click cycles: denoised → raw → normals → depth → NRD validation.
    let mut blit_view_mode: u32 = 0;

    'running: loop {
        let mut pump = ctx.event_pump().unwrap();
        for event in pump.poll_iter() {
            match event {
                sdl3::event::Event::Quit { .. } => break 'running,
                sdl3::event::Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    blit_view_mode = (blit_view_mode + 1) % 5;
                    let label = match blit_view_mode {
                        0 => "denoised",
                        1 => "raw (no denoiser)",
                        2 => "normals",
                        3 => "primary hit distance",
                        4 => "NRD validation (OUT_VALIDATION)",
                        _ => "?",
                    };
                    eprintln!("viewport: {label}");
                }
                _ => {}
            }
        }

        let wgpu::CurrentSurfaceTexture::Success(surface_tex) = surface.get_current_texture()
        else {
            continue;
        };
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let eye = Vec3::new(0.5, 0.5, -1.25);
        let at = Vec3::new(0.5, 0.5, 0.5);
        let view = Mat4::look_at_rh(eye, at, Vec3::Y);
        let proj = Mat4::perspective_rh(
            50.0_f32.to_radians(),
            surface_config.width as f32 / surface_config.height as f32,
            0.01,
            100.0,
        );
        let u = RtUniforms {
            view_inv: view.inverse().to_cols_array(),
            proj_inv: proj.inverse().to_cols_array(),
            view: view.to_cols_array(),
            frame_data: [frame_index, 0, 0, 0],
            hit_dist_and_rough: [
                reblur_for_pipelines.hitDistanceParameters.A,
                reblur_for_pipelines.hitDistanceParameters.B,
                reblur_for_pipelines.hitDistanceParameters.C,
                0.6,
            ],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(
            &blit_uniform_buf,
            0,
            bytemuck::bytes_of(&BlitUniforms {
                mode: blit_view_mode,
                _pad: [0; 3],
            }),
        );

        let mut common_settings = default_common_settings();
        common_settings.resourceSize = [surface_config.width as u16, surface_config.height as u16];
        common_settings.resourceSizePrev =
            [surface_config.width as u16, surface_config.height as u16];
        common_settings.rectSize = [surface_config.width as u16, surface_config.height as u16];
        common_settings.rectSizePrev = [surface_config.width as u16, surface_config.height as u16];
        common_settings.viewZScale = 1.0;
        common_settings.frameIndex = frame_index;
        common_settings.timeDeltaBetweenFrames = 1.0 / 60.0;
        common_settings.worldToViewMatrix = view.to_cols_array();
        common_settings.worldToViewMatrixPrev = view.to_cols_array();
        common_settings.viewToClipMatrix = proj.to_cols_array();
        common_settings.viewToClipMatrixPrev = proj.to_cols_array();
        common_settings.enableValidation = blit_view_mode == 4;
        wgpu_nrd
            .instance
            .set_common_settings(&common_settings)
            .unwrap();

        let mut user_resources = UserResources::default();
        user_resources
            .by_type
            .insert(ResourceType::InMv as u32, v_in_mv.clone());
        user_resources.split_sampled_storage.insert(
            ResourceType::InViewz as u32,
            (v_in_view_z.clone(), v_in_view_z_storage.clone()),
        );
        user_resources
            .by_type
            .insert(ResourceType::InNormalRoughness as u32, v_in_nr.clone());
        user_resources.by_type.insert(
            ResourceType::InDiffRadianceHitdist as u32,
            v_in_diff.clone(),
        );
        user_resources.by_type.insert(
            ResourceType::OutDiffRadianceHitdist as u32,
            v_out_diff.clone(),
        );
        if common_settings.enableValidation {
            user_resources
                .by_type
                .insert(ResourceType::OutValidation as u32, v_out_validation.clone());
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let gw = (surface_config.width + 7) / 8;
            let gh = (surface_config.height + 7) / 8;
            cpass.set_pipeline(&gbuffer_pipeline);
            cpass.set_bind_group(0, &gbuffer_bind_group, &[]);
            cpass.dispatch_workgroups(gw, gh, 1);
            cpass.set_pipeline(&diffuse_pipeline);
            cpass.set_bind_group(0, &diffuse_bind_group, &[]);
            cpass.dispatch_workgroups(gw, gh, 1);
        }

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &in_view_z,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &in_view_z_storage,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            gbuffer_extent,
        );

        if blit_view_mode == 0 || blit_view_mode == 4 {
            wgpu_nrd
                .encode_dispatches(
                    &mut encoder,
                    &device,
                    &queue,
                    &[denoisers[0].identifier],
                    &user_resources,
                )
                .unwrap();
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&blit_pipeline);
            rpass.set_bind_group(0, &blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        surface_tex.present();
        frame_index = frame_index.wrapping_add(1);
    }
}
