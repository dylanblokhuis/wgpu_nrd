mod geometry;
mod pipelines;
mod scene;

use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use pipelines::{CornellRenderPipelines, FrameTextureViews};
use scene::CornellScene;
use sdl3::mouse::MouseButton;
use wgpu::SurfaceTargetUnsafe;
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
struct BlitUniforms {
    mode: u32,
    _pad: [u32; 3],
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
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        // apply_limit_buckets: false,
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::EXPERIMENTAL_RAY_QUERY
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
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: cap.formats[0],
        view_formats: vec![cap.formats[0].add_srgb_suffix()],
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        width,
        height,
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

    let scene = CornellScene::new(&device);

    let pipelines = CornellRenderPipelines::new(&device, &mut wesl, surface_config.format);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("as_build"),
    });
    scene.encode_build(&mut encoder);
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

    let frame_views = FrameTextureViews {
        v_in_mv: v_in_mv.clone(),
        v_in_view_z: v_in_view_z.clone(),
        v_in_nr: v_in_nr.clone(),
        v_in_albedo: v_in_albedo.clone(),
        v_in_emissive: v_in_emissive.clone(),
        v_in_diff: v_in_diff.clone(),
        v_out_diff: v_out_diff.clone(),
        v_out_validation: v_out_validation.clone(),
    };
    let bind_groups = pipelines.create_bind_groups(
        &device,
        &scene,
        &uniform_buf,
        &blit_uniform_buf,
        &frame_views,
    );

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
            cpass.set_pipeline(&pipelines.gbuffer);
            cpass.set_bind_group(0, &bind_groups.gbuffer, &[]);
            cpass.dispatch_workgroups(gw, gh, 1);
            cpass.set_pipeline(&pipelines.diffuse);
            cpass.set_bind_group(0, &bind_groups.diffuse, &[]);
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
            rpass.set_pipeline(&pipelines.blit);
            rpass.set_bind_group(0, &bind_groups.blit, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        surface_tex.present();
        frame_index = frame_index.wrapping_add(1);
    }
}
