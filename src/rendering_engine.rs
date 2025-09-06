use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::util::*;

// use crevice

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt}, BufferUsages, PollType, ShaderStages, TextureUsages
};
use winit::window::Window;

pub enum SurfaceSource {
    Window(Arc<Window>),
    Standalone(u32, u32),
}

pub enum SurfaceDest {
    Window {
        window: Arc<Window>,
        format: wgpu::TextureFormat,
        surface: wgpu::Surface<'static>,
    },
    Standalone {
        format: wgpu::TextureFormat,
        size: (u32, u32),
        texture: wgpu::Texture,
        readback_buffer: wgpu::Buffer,
    }
}

impl SurfaceDest {
    pub fn get_format(&self) -> wgpu::TextureFormat {
        match self {
            &SurfaceDest::Window { format, .. } | &SurfaceDest::Standalone { format, .. } => format,
        }
    }
    
    pub fn get_handle(&self) -> SurfaceHandle {
        match self {
            &Self::Window { format, ref surface, .. } => {
                let surface_texture = surface.get_current_texture().expect("could not get window surface's texture");
                let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(format.add_srgb_suffix()),
                    ..default()
                });
                SurfaceHandle::Window(surface_texture, view)
            },
            &Self::Standalone { format, ref texture, .. } => {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(format),
                    ..default()
                });
                SurfaceHandle::Standalone(view)
            }
        }
    }
    
    pub fn queue_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Window { .. } => {},
            Self::Standalone { texture, readback_buffer, .. } => {
                let size = texture.size();
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some((size.width * 4) as u32),
                            rows_per_image: Some(size.height as u32),
                        },
                    },
                    size,
                );
            },
        }
    }
    
    pub fn pre_present(&self) {
        match self {
            Self::Window { window, .. } => window.pre_present_notify(),
            Self::Standalone { .. } => {},
        }
    }
}

pub enum SurfaceHandle {
    Window(wgpu::SurfaceTexture, wgpu::TextureView),
    Standalone(wgpu::TextureView),
}

impl SurfaceHandle {
    pub fn get_view(&self) -> &wgpu::TextureView {
        match self {
            SurfaceHandle::Window(_, texture_view) | SurfaceHandle::Standalone(texture_view) => texture_view,
        }
    }
    
    pub fn present(self) {
        match self {
            Self::Window(surface, _) => {
                surface.present();
            },
            Self::Standalone(_) => {
                // TODO: maybe write to a .png?
            },
        }
    }
}

pub struct RenderingEngine {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub surface_dest: SurfaceDest,
    pub render_data: RenderDataRef,
    pub render_settings: RenderSettings,
    pub pipelines: Pipelines,
}

impl RenderingEngine {
    pub async fn new(
        surface_source: SurfaceSource,
        init_render_data: RenderData,
        init_render_settings: RenderSettings,
    ) -> Self {
        let instance = wgpu::Instance::new(&default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..default()
            })
            .await
            .expect("could not request gpu adapter");
        eprintln!("adapter info: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features {
                    features_wgpu: wgpu::FeaturesWGPU::PUSH_CONSTANTS,
                    ..default()
                },
                required_limits: wgpu::Limits {
                    max_push_constant_size: 64,
                    ..default()
                },
                ..default()
            })
            .await
            .expect("could not request device");

        let surface_dest = match surface_source {
            SurfaceSource::Window(window) => {
                let size = window.inner_size();
                let surface = instance
                    .create_surface(window.clone())
                    .expect("could not create gpu surface");
                let format = surface.get_capabilities(&adapter).formats[0];
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                    view_formats: vec![format.add_srgb_suffix()],
                };
                surface.configure(&device, &config);
                SurfaceDest::Window {
                    window,
                    format,
                    surface,
                }
            }
            SurfaceSource::Standalone(width, height) => {
                let format = wgpu::TextureFormat::Rgba8Unorm;
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                    view_formats: &[format],
                });
                assert_eq!(format.block_dimensions(), (1, 1));
                let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (format.block_copy_size(None).unwrap() * width * height) as _,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                SurfaceDest::Standalone {
                    format,
                    size: (width, height),
                    texture,
                    readback_buffer,
                }
            }
        };

        let pipelines = Pipelines::new(surface_dest.get_format(), &device, &queue);
        Self {
            surface_dest,
            instance,
            adapter,
            device,
            queue,
            render_data: Rc::new(RefCell::new(init_render_data)),
            render_settings: init_render_settings,
            pipelines,
        }
    }
    
    pub async fn readback_pixels(&self) -> Vec<u8> {
        match &self.surface_dest {
            SurfaceDest::Window { .. } => unreachable!("trying to readback from window surface"),
            &SurfaceDest::Standalone { ref texture, ref readback_buffer, size: (width, height), .. } => {
                let slice = readback_buffer.slice(..);
                let (sender, receiver) = flume::bounded(1);
                slice.map_async(wgpu::MapMode::Read, move |res| sender.send(res).unwrap());
                self.device.poll(wgpu::PollType::Wait).unwrap();
                receiver.recv_async().await.unwrap().unwrap();
                
                let view = slice.get_mapped_range();
                let mut pixels = Vec::with_capacity(width as usize * height as usize * size_of::<u32>());
                pixels.extend_from_slice(&view);
                drop(view);
                readback_buffer.unmap();
                
                pixels
            },
        }
    }
}

pub struct Pipelines {
    pub render_pipeline: wgpu::RenderPipeline,
    
    /*pub compute: wgpu::ComputePipeline,
    // rasterizer: wgpu::ComputePipeline,
    pub blitting: wgpu::RenderPipeline,

    pub histogram_bind_group: wgpu::BindGroup,
    pub histogram_buffer: wgpu::Buffer,

    pub render_data_bind_group: wgpu::BindGroup,
    pub variables_buffer: wgpu::Buffer,
    pub transform_buffer: wgpu::Buffer,
    pub weights_buffer: wgpu::Buffer,

    pub render_settings_bind_group: wgpu::BindGroup,
    pub render_settings_buffer: wgpu::Buffer,*/

}

impl Pipelines {
    pub fn new(surface_format: wgpu::TextureFormat, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        /*// =======
        // buffers
        // =======

        let mut buf_init = vec![0u8; (args.width * args.height * size_of::<u32>() as u32) as usize];

        // =======
        // Read only
        // =======

        //Metadata buffer
        buf_init.clear();
        buf_init.resize(size_of::<u32>(), 0);
        let metadata_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE | BufferUsages::UNIFORM,
            contents: &buf_init,
        });
        //====================================================================================

        //entropy buffer =======================================================================
        buf_init.clear();
        let num_entroy_samples = 1024;
        let mut rng = rand::rng();
        for _ in 0..num_entroy_samples {
            let sample: u32 = rng.random();
            buf_init.extend(sample.to_ne_bytes());
        }
        let entropy_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE,
            contents: &buf_init,
        });
        //============================================================================================

        //variables buffer
        //TODO
        //====================================================================================

        //transform buffer
        //
        //====================================================================================

        // =======
        // Read/Write
        // =======

        //IfsHistogramData Buffer ====================================================================

        buf_init.clear();
        let samples_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE,
            contents: &buf_init,
        });
        //====================================================================================

        //Init/final values buffer============================================================
        //TODO
        //====================================================================================

        // ==================
        // IfsHistogramData bind group
        // ==================
        let samples_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },

                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },

                        count: None,
                    },
                ],
            });
        let samples_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &samples_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &samples_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &entropy_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // ===================
        // metadata bind group
        // ===================
        let metadata_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let metadata_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &metadata_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &metadata_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // ===================
        // chaos game pipeline
        // ===================
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("chaos_game.wgsl"))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&samples_group_layout, &metadata_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..size_of::<(u32, u32, f32, u32)>() as _,
            }],
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: default(),
            cache: default(),
        });

        // ===============
        // render pipeline
        // ===============
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&samples_group_layout, &metadata_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::FRAGMENT,
                range: 0..size_of::<(u32, u32, f32, u32)>() as _,
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quad.wgsl"))),
        });
        let blitting = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: default(),
                targets: &[Some(surface_format.into())],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            multiview: None,
            cache: None,
        });*/
        
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("triangle.wgsl").into()),
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                compilation_options: default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                compilation_options: default(),
                targets: &[Some(surface_format.into())],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            multiview: None,
            cache: None,
        });
        
        Self {
            render_pipeline,
            /*compute,
            blitting,
            histogram_bind_group: todo!(),
            histogram_buffer: todo!(),
            render_data_bind_group: todo!(),
            variables_buffer: todo!(),
            transform_buffer: todo!(),
            weights_buffer: todo!(),
            render_settings_bind_group: todo!(),
            render_settings_buffer: todo!(),*/
        }
    }
}