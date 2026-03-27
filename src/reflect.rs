//! SPIR-V reflection via [`rspirv_reflect`] (workgroup size + bind group layout hints).
//!
//! We avoid [`naga`]'s SPIR-V front end: it rejects capabilities NRD enables (for example
//! `StorageImageWriteWithoutFormat`) even when shaders are only passed through to the driver.

use crate::WgpuNrdError;
use crate::format::wgpu_texture_binding_sample_type;
use rspirv_reflect::rspirv::dr::{Instruction, Module, Operand};
use rspirv_reflect::{DescriptorInfo, DescriptorType, ReflectError, Reflection, spirv};

pub use rspirv_reflect;

/// Parse SPIR-V bytes for reflection (descriptor sets + execution mode).
pub fn parse_spirv(spirv: &[u8]) -> Result<Reflection, WgpuNrdError> {
    Reflection::new_from_spirv(spirv)
        .map_err(|e: ReflectError| WgpuNrdError::SpirvReflect(e.to_string()))
}

/// Local workgroup size for the compute entry named `entry_name`.
pub fn compute_workgroup_size(
    reflection: &Reflection,
    entry_name: &str,
) -> Result<(u32, u32, u32), WgpuNrdError> {
    let m = &reflection.0;
    let mut entry_point_id: Option<u32> = None;
    for inst in m.global_inst_iter() {
        if inst.class.opcode != spirv::Op::EntryPoint {
            continue;
        }
        let em = match inst.operands.first() {
            Some(Operand::ExecutionModel(em)) => *em,
            _ => continue,
        };
        if em != spirv::ExecutionModel::GLCompute {
            continue;
        }
        let eid = match inst.operands.get(1) {
            Some(Operand::IdRef(id)) => *id,
            _ => {
                return Err(WgpuNrdError::SpirvReflect(
                    "OpEntryPoint: expected entry point id".into(),
                ));
            }
        };
        let name = match inst.operands.get(2) {
            Some(Operand::LiteralString(s)) => s.as_str(),
            _ => {
                return Err(WgpuNrdError::SpirvReflect(
                    "OpEntryPoint: expected entry point name".into(),
                ));
            }
        };
        if name == entry_name {
            entry_point_id = Some(eid);
            break;
        }
    }
    let ep =
        entry_point_id.ok_or_else(|| WgpuNrdError::EntryPointNotFound(entry_name.to_string()))?;

    for inst in m.global_inst_iter() {
        if inst.class.opcode != spirv::Op::ExecutionMode {
            continue;
        }
        let id = match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        if id != ep {
            continue;
        }
        let mode = match inst.operands.get(1) {
            Some(Operand::ExecutionMode(m)) => *m,
            _ => continue,
        };
        if !matches!(
            mode,
            spirv::ExecutionMode::LocalSize | spirv::ExecutionMode::LocalSizeHint
        ) {
            continue;
        }
        let x = match inst.operands.get(2) {
            Some(Operand::LiteralBit32(v)) => *v,
            _ => continue,
        };
        let y = match inst.operands.get(3) {
            Some(Operand::LiteralBit32(v)) => *v,
            _ => continue,
        };
        let z = match inst.operands.get(4) {
            Some(Operand::LiteralBit32(v)) => *v,
            _ => continue,
        };
        return Ok((x, y, z));
    }

    Err(WgpuNrdError::SpirvReflect(format!(
        "no OpExecutionMode LocalSize for entry {entry_name:?}"
    )))
}

fn find_assignment(module: &Module, id: u32) -> Option<&Instruction> {
    module
        .types_global_values
        .iter()
        .find(|i| i.result_id == Some(id))
}

/// Map SPIR-V image format to wgpu. `Unknown` is common with
/// `StorageImageWriteWithoutFormat`; layout must match the texture views you bind.
pub fn spirv_image_format_to_wgpu(fmt: spirv::ImageFormat) -> wgpu::TextureFormat {
    use spirv::ImageFormat as F;
    use wgpu::TextureFormat as T;
    match fmt {
        F::Unknown => T::Rgba16Float,
        F::Rgba32f => T::Rgba32Float,
        F::Rgba16f => T::Rgba16Float,
        F::R32f => T::R32Float,
        F::Rgba8 => T::Rgba8Unorm,
        F::Rgba8Snorm => T::Rgba8Snorm,
        F::Rg32f => T::Rg32Float,
        F::Rg16f => T::Rg16Float,
        F::R11fG11fB10f => T::Rg11b10Ufloat,
        F::R16f => T::R16Float,
        F::Rgba16 => T::Rgba16Unorm,
        F::Rgb10A2 => T::Rgb10a2Unorm,
        F::Rg16 => T::Rg16Unorm,
        F::Rg8 => T::Rg8Unorm,
        F::R16 => T::R16Unorm,
        F::R8 => T::R8Unorm,
        F::Rgba16Snorm => T::Rgba16Snorm,
        F::Rg16Snorm => T::Rg16Snorm,
        F::Rg8Snorm => T::Rg8Snorm,
        F::R16Snorm => T::R16Snorm,
        F::R8Snorm => T::R8Snorm,
        F::Rgba32i => T::Rgba32Sint,
        F::Rgba16i => T::Rgba16Sint,
        F::Rgba8i => T::Rgba8Sint,
        F::R32i => T::R32Sint,
        F::Rg32i => T::Rg32Sint,
        F::Rg16i => T::Rg16Sint,
        F::Rg8i => T::Rg8Sint,
        F::R16i => T::R16Sint,
        F::R8i => T::R8Sint,
        F::Rgba32ui => T::Rgba32Uint,
        F::Rgba16ui => T::Rgba16Uint,
        F::Rgba8ui => T::Rgba8Uint,
        F::R32ui => T::R32Uint,
        F::Rgb10a2ui => T::Rgb10a2Uint,
        F::Rg32ui => T::Rg32Uint,
        F::Rg16ui => T::Rg16Uint,
        F::Rg8ui => T::Rg8Uint,
        F::R16ui => T::R16Uint,
        F::R8ui => T::R8Uint,
        F::R64ui | F::R64i => T::Rgba32Uint,
    }
}

fn dim_to_wgpu(dim: spirv::Dim) -> wgpu::TextureViewDimension {
    match dim {
        spirv::Dim::Dim1D | spirv::Dim::DimBuffer => wgpu::TextureViewDimension::D1,
        spirv::Dim::Dim2D => wgpu::TextureViewDimension::D2,
        spirv::Dim::Dim3D => wgpu::TextureViewDimension::D3,
        spirv::Dim::DimCube => wgpu::TextureViewDimension::Cube,
        spirv::Dim::DimRect | spirv::Dim::DimSubpassData | spirv::Dim::DimTileImageDataEXT => {
            wgpu::TextureViewDimension::D2
        }
    }
}

/// Always `WriteOnly`: many backends (notably Metal) reject `ReadWrite` for several float storage
/// formats (e.g. `Rg16Float`) even when SPIR-V omits an access qualifier or uses RW. NRD passthrough
/// shaders still match; layout must satisfy `create_bind_group_layout` on the target adapter.
fn storage_texture_access(_ty: &Instruction) -> Result<wgpu::StorageTextureAccess, WgpuNrdError> {
    Ok(wgpu::StorageTextureAccess::WriteOnly)
}

fn resolve_type_image<'a>(
    module: &'a Module,
    mut type_id: u32,
) -> Result<&'a Instruction, WgpuNrdError> {
    loop {
        let ty = find_assignment(module, type_id).ok_or_else(|| {
            WgpuNrdError::SpirvReflect(format!("missing SPIR-V type id {type_id}"))
        })?;
        match ty.class.opcode {
            spirv::Op::TypePointer => {
                type_id = operand_id_ref(ty, 1)?;
            }
            spirv::Op::TypeArray | spirv::Op::TypeRuntimeArray => {
                type_id = operand_id_ref(ty, 0)?;
            }
            spirv::Op::TypeSampledImage => {
                type_id = operand_id_ref(ty, 0)?;
            }
            spirv::Op::TypeImage => return Ok(ty),
            _ => {
                return Err(WgpuNrdError::SpirvReflect(format!(
                    "expected image-related type, got {:?}",
                    ty.class.opcode
                )));
            }
        }
    }
}

fn operand_id_ref(inst: &Instruction, idx: usize) -> Result<u32, WgpuNrdError> {
    match inst.operands.get(idx) {
        Some(Operand::IdRef(id)) => Ok(*id),
        _ => Err(WgpuNrdError::SpirvReflect(format!(
            "operand {idx} not IdRef for {:?}",
            inst.class.opcode
        ))),
    }
}

fn operand_dim(inst: &Instruction) -> Result<spirv::Dim, WgpuNrdError> {
    match inst.operands.get(1) {
        Some(Operand::Dim(d)) => Ok(*d),
        _ => Err(WgpuNrdError::SpirvReflect(
            "OpTypeImage: missing Dim".into(),
        )),
    }
}

fn operand_image_format(inst: &Instruction) -> spirv::ImageFormat {
    match inst.operands.get(6) {
        Some(Operand::ImageFormat(f)) => *f,
        _ => spirv::ImageFormat::Unknown,
    }
}

fn operand_multisampled(inst: &Instruction) -> bool {
    matches!(inst.operands.get(4), Some(Operand::LiteralBit32(1)))
}

fn sampled_texture_entry(
    module: &Module,
    group: u32,
    binding: u32,
    info: &DescriptorInfo,
) -> Result<wgpu::BindGroupLayoutEntry, WgpuNrdError> {
    let type_id = find_var_result_type(module, group, binding).ok_or_else(|| {
        WgpuNrdError::SpirvReflect(format!(
            "missing variable for set {group} binding {binding}"
        ))
    })?;
    let image_ty = resolve_type_image(module, type_id)?;
    let view_dim = dim_to_wgpu(operand_dim(image_ty)?);
    let multisampled = operand_multisampled(image_ty);
    let spirv_fmt = operand_image_format(image_ty);
    let wgpu_fmt = spirv_image_format_to_wgpu(spirv_fmt);
    let depth_image = matches!(image_ty.operands.get(2), Some(Operand::LiteralBit32(1)));
    let sample_type = wgpu_texture_binding_sample_type(wgpu_fmt).unwrap_or_else(|| {
        if depth_image {
            wgpu::TextureSampleType::Depth
        } else {
            wgpu::TextureSampleType::Float { filterable: true }
        }
    });
    Ok(wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: view_dim,
            multisampled,
        },
        count: binding_count_wgpu(&info.binding_count),
    })
}

/// Build [`wgpu::BindGroupLayoutEntry`] list for descriptor set `group` from reflected SPIR-V.
pub fn bind_group_layout_entries(
    reflection: &Reflection,
    group: u32,
    min_uniform_binding_size: u32,
) -> Result<Vec<wgpu::BindGroupLayoutEntry>, WgpuNrdError> {
    let sets = reflection
        .get_descriptor_sets()
        .map_err(|e| WgpuNrdError::SpirvReflect(e.to_string()))?;
    let Some(bindings) = sets.get(&group) else {
        return Ok(Vec::new());
    };

    let mut out: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
    let ubo_size = min_uniform_binding_size.max(1) as u64;
    let module = &reflection.0;

    for (&binding, info) in bindings.iter() {
        let entry = match info.ty {
            DescriptorType::UNIFORM_BUFFER => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(ubo_size),
                },
                count: binding_count_wgpu(&info.binding_count),
            },
            DescriptorType::STORAGE_BUFFER => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: binding_count_wgpu(&info.binding_count),
            },
            DescriptorType::SAMPLER => wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: binding_count_wgpu(&info.binding_count),
            },
            DescriptorType::SAMPLED_IMAGE | DescriptorType::COMBINED_IMAGE_SAMPLER => {
                sampled_texture_entry(module, group, binding, info)?
            }
            DescriptorType::STORAGE_IMAGE => {
                let image_ty = {
                    let type_id =
                        find_var_result_type(module, group, binding).ok_or_else(|| {
                            WgpuNrdError::SpirvReflect(format!(
                                "missing variable for set {group} binding {binding}"
                            ))
                        })?;
                    resolve_type_image(module, type_id)?
                };
                let dim = operand_dim(image_ty)?;
                let fmt = spirv_image_format_to_wgpu(operand_image_format(image_ty));
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: storage_texture_access(image_ty)?,
                        format: fmt,
                        view_dimension: dim_to_wgpu(dim),
                    },
                    count: binding_count_wgpu(&info.binding_count),
                }
            }
            DescriptorType::UNIFORM_TEXEL_BUFFER | DescriptorType::STORAGE_TEXEL_BUFFER => {
                return Err(WgpuNrdError::SpirvReflect(
                    "texel buffer bindings are not supported yet".into(),
                ));
            }
            _ => {
                return Err(WgpuNrdError::SpirvReflect(format!(
                    "unsupported descriptor type {:?} at set {group} binding {binding}",
                    info.ty
                )));
            }
        };
        out.push(entry);
    }

    out.sort_by_key(|e| e.binding);
    Ok(out)
}

fn find_var_result_type(module: &Module, set: u32, binding: u32) -> Option<u32> {
    for var in module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == spirv::Op::Variable)
    {
        let cls = match var.operands.first() {
            Some(Operand::StorageClass(c)) => *c,
            _ => continue,
        };
        if !matches!(
            cls,
            spirv::StorageClass::Uniform
                | spirv::StorageClass::UniformConstant
                | spirv::StorageClass::StorageBuffer
        ) {
            continue;
        }
        let vid = var.result_id?;
        let mut vset = None;
        let mut vbind = None;
        for ann in module
            .annotations
            .iter()
            .filter(|a| a.class.opcode == spirv::Op::Decorate)
        {
            if ann.operands.first() != Some(&Operand::IdRef(vid)) {
                continue;
            }
            if let (
                Some(Operand::Decoration(spirv::Decoration::DescriptorSet)),
                Some(Operand::LiteralBit32(s)),
            ) = (ann.operands.get(1), ann.operands.get(2))
            {
                vset = Some(*s);
            }
            if let (
                Some(Operand::Decoration(spirv::Decoration::Binding)),
                Some(Operand::LiteralBit32(b)),
            ) = (ann.operands.get(1), ann.operands.get(2))
            {
                vbind = Some(*b);
            }
        }
        if vset == Some(set) && vbind == Some(binding) {
            return var.result_type;
        }
    }
    None
}

fn binding_count_wgpu(bc: &rspirv_reflect::BindingCount) -> Option<std::num::NonZeroU32> {
    match bc {
        rspirv_reflect::BindingCount::One => None,
        rspirv_reflect::BindingCount::StaticSized(n) => std::num::NonZeroU32::new(*n as u32),
        rspirv_reflect::BindingCount::Unbounded => None,
    }
}
