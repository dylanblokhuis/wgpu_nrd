//! Cornell box scene: mesh, materials, and acceleration structures for ray tracing.

use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::Affine3A;
use wgpu::util::DeviceExt;

use crate::geometry::{BlasVertex, build_cornell_mesh};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub albedo: [f32; 4],
    pub emissive: [f32; 4],
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

pub struct CornellScene {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub tri_meta_buf: wgpu::Buffer,
    pub materials_buf: wgpu::Buffer,
    pub blas: wgpu::Blas,
    pub tlas: wgpu::Tlas,
    blas_geo_size_desc: wgpu::BlasTriangleGeometrySizeDescriptor,
}

impl CornellScene {
    pub fn new(device: &wgpu::Device) -> Self {
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

        Self {
            vertex_buf,
            index_buf,
            tri_meta_buf,
            materials_buf,
            blas,
            tlas,
            blas_geo_size_desc,
        }
    }

    pub fn encode_build(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.build_acceleration_structures(
            std::iter::once(&wgpu::BlasBuildEntry {
                blas: &self.blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &self.blas_geo_size_desc,
                    vertex_buffer: &self.vertex_buf,
                    first_vertex: 0,
                    vertex_stride: mem::size_of::<BlasVertex>() as u64,
                    index_buffer: Some(&self.index_buf),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            std::iter::once(&self.tlas),
        );
    }
}
