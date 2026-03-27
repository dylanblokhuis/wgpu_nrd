# wgpu_nrd

**wgpu** integration for
[NVIDIA NRD](https://github.com/NVIDIAGameWorks/RayTracingDenoiser)

Still work in progress, the rusty_nrd api needs to be cleaned up.

## What it does

- Builds an NRD `Instance` from your `DenoiserSlot` configuration and wires up
  pool textures, samplers, constant buffer, and compute pipelines.
- Loads NRD shaders with
  [`wgpu::Device::create_shader_module_passthrough`](https://docs.rs/wgpu/latest/wgpu/struct.Device.html#method.create_shader_module_passthrough)
  (SPIR-V, DXIL, or precompiled Metal as provided by NRD).
- Exposes NRD helper functions with
  [WESL](https://github.com/wgsl-tooling-wg/wesl-rs). See the `cornell-demo` on
  how its been used.

Use `WgpuNrd::new` to construct, then `encode_dispatches` after updating
denoiser settings on the instance.

## Example

```sh
cargo run -p cornell-demo
```

## Third-party / NVIDIA NRD

Rust code in this repository is **MIT OR Apache-2.0**. The **NVIDIA NRD** (and
other NVIDIA RTX SDK) binaries, shaders, and headers you obtain from NVIDIA are
governed separately — see **`NOTICE`**

## License

Dual-licensed under **MIT OR Apache-2.0**, matching the crate metadata.
