#![allow(dead_code, unused_imports, unused_variables)]

use std::{borrow::Cow, ops::Deref, sync::Arc, time::Instant};

use clap::Parser;
use rand::Rng;
use wgpu::{util::{BufferInitDescriptor, DeviceExt}, BufferUsages, PollType, ShaderStages};
use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::{ElementState, KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{Key, NamedKey}, window::Window};

#[derive(Debug, Parser)]
#[command(disable_help_flag = true)]
struct AppArgs {
    #[arg(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,

    #[arg(short, long, default_value_t = 512)]
    width: u32,

    #[arg(short, long, default_value_t = 512)]
    height: u32,
}

fn main() -> anyhow::Result<()> {
    let args = AppArgs::parse();
    #[cfg(debug_assertions)]
    dbg!(&args);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[derive(Default)]
struct App {
    args: Option<AppArgs>,
    state: Option<AppState>,
}

impl App {
    fn new(args: AppArgs) -> Self {
        Self {
            args: Some(args),
            state: None
        }
    }
}

impl Deref for App {
    type Target = AppState;

    fn deref(&self) -> &Self::Target {
        &self.state.as_ref().unwrap()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let args = self.args.take().expect("app missing its arg field");

        let mut window_attrs = Window::default_attributes();
        window_attrs.title = "Warp Speed Chaos Game".into();
        window_attrs.resizable = false; // TODO: requires recreating framebuffer
        let window_size = PhysicalSize {
            width: args.width,
            height: args.height,
        };
        window_attrs.inner_size = Some(window_size.into());

        let window = event_loop.create_window(window_attrs).unwrap();
        if let Some(primary_monitor) = event_loop.primary_monitor() {
            let monitor_size = primary_monitor.size();
            let position = winit::dpi::PhysicalPosition {
                x: (monitor_size.width.saturating_sub(window_size.width)) / 2,
                y: (monitor_size.height.saturating_sub(window_size.width)) / 2,
            };
            window.set_outer_position(position);
        } else {
            eprintln!("couldn't determine primary monitor, can't set window position :(");
        }
        window.request_redraw();

        let state = pollster::block_on(async { AppState::new(args, window).await });
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                event_loop.exit();
            },
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key,
                    state: key_state,
                    repeat,
                    ..
                },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Escape) => {
                    event_loop.exit();
                },
                _ if !repeat && key_state == ElementState::Pressed => {
                    self.window.request_redraw();
                },
                _ => {},
            },
            WindowEvent::RedrawRequested => {
                // self.window.request_redraw();
                self.render();
            },

            // just to hush logging
            WindowEvent::CursorMoved { .. } |
            WindowEvent::Focused(..) |
            WindowEvent::Moved(..) |
            WindowEvent::CursorEntered { .. } |
            WindowEvent::CursorLeft { .. } |
            WindowEvent::ModifiersChanged(..) |
            WindowEvent::MouseInput { .. } |
            WindowEvent::MouseWheel { .. } => {},
            _ => eprintln!("unknown window event {event:#?}"),
        }
    }
}

struct AppState {
    args: AppArgs,
    window: Arc<Window>,
    epoch: Instant,

    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    surface: wgpu::Surface<'static>,

    pipelines: Pipelines,
}

struct Pipelines {
    compute: wgpu::ComputePipeline,
    // rasterizer: wgpu::ComputePipeline,
    blitting: wgpu::RenderPipeline,

    samples_bind_group: wgpu::BindGroup,
    samples_buffer: wgpu::Buffer,

    metadata_bind_group: wgpu::BindGroup,
    metadata_buffer: wgpu::Buffer,
}

impl AppState {
    async fn new(args: AppArgs, window: Window) -> Self {
        let window = window.into();
        let epoch = Instant::now();
        let (device, queue, surface_format, surface) = Self::init_wgpu(&window).await;
        let pipelines = Self::init_pipelines(&args, &device, surface_format).await;
        Self {
            args,
            window,
            epoch,

            device,
            queue,
            surface_format,
            surface,

            pipelines,
        }
    }

    async fn init_wgpu(window: &Arc<Window>) -> (wgpu::Device, wgpu::Queue, wgpu::TextureFormat, wgpu::Surface<'static>) {
        let instance = wgpu::Instance::new(&default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..default()
        }).await.expect("could not request gpu adapter");
        eprintln!("adapter info: {:?}", adapter.get_info());

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features {
                features_wgpu: wgpu::FeaturesWGPU::PUSH_CONSTANTS,
                ..default()
            },
            required_limits: wgpu::Limits {
                max_push_constant_size: 64,
                ..default()
            },
            ..default()
        }).await.expect("could not request device");

        let window_size = window.inner_size();
        let surface = instance.create_surface(window.clone()).expect("could not create gpu surface");
        let surface_format = surface.get_capabilities(&adapter).formats[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![surface_format.add_srgb_suffix()],
        };
        surface.configure(&device, &surface_config);

        (device, queue, surface_format, surface)
    }

    async fn init_pipelines(args: &AppArgs, device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Pipelines {
        // =======
        // buffers
        // =======
        let mut buf_init = vec![0u8; (args.width * args.height * size_of::<u32>() as u32) as usize];
        let samples_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE,
            contents: &buf_init,
        });

        buf_init.resize(size_of::<u32>(), 0);
        let metadata_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE | BufferUsages::UNIFORM,
            contents: &buf_init,
        });

        let num_entroy_samples = 1024;
        let mut rng = rand::rng();
        buf_init.clear();
        for _ in 0 .. num_entroy_samples {
            let sample: u32 = rng.random();
            buf_init.extend(sample.to_ne_bytes());
        }
        let entropy_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE,
            contents: &buf_init,
        });



        // ==================
        // samples bind group
        // ==================
        let samples_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            }],
        });
        let samples_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &samples_group_layout,
            entries: &[wgpu::BindGroupEntry {
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
            }
            ],
        });

        // ===================
        // metadata bind group
        // ===================
        let metadata_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                range: 0 .. size_of::<(u32, u32, f32, u32)>() as _,
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
                range: 0 .. size_of::<(u32, u32, f32, u32)>() as _,
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
        });

        Pipelines {
            compute,
            blitting,

            samples_bind_group,
            samples_buffer,

            metadata_bind_group,
            metadata_buffer,
        }
    }

    fn render(&self) {
        let now = Instant::now();
        let iters_per_invocation = 1000u32;
        // FIXME: pass 64 bit time
        let now_secs = (now - self.epoch).as_secs_f32();

        let mut pc_buf = [0u8; size_of::<(u32, u32, f32, u32)>()];
        (&mut pc_buf[0 .. 4]).copy_from_slice(&self.args.width.to_ne_bytes());
        (&mut pc_buf[4 .. 8]).copy_from_slice(&self.args.height.to_ne_bytes());
        (&mut pc_buf[8 .. 12]).copy_from_slice(&now_secs.to_ne_bytes());
        (&mut pc_buf[12 .. 16]).copy_from_slice(&iters_per_invocation.to_ne_bytes());

        let surface_texture = self.surface.get_current_texture().expect("could not get surface's texture");
        let texture_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.surface_format.add_srgb_suffix()),
            ..default()
        });

        let mut encoder = self.device.create_command_encoder(&default());

        for _ in 0 .. 1 {
            let now = Instant::now();
            let now_secs = (now - self.epoch).as_secs_f32();
            (&mut pc_buf[8 .. 12]).copy_from_slice(&now_secs.to_ne_bytes());

            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&self.pipelines.compute);
            pass.set_bind_group(0, &self.pipelines.samples_bind_group, &[]);
            pass.set_bind_group(1, &self.pipelines.metadata_bind_group, &[]);
            pass.set_push_constants(0, &pc_buf);
            pass.dispatch_workgroups(16384, 1 , 1);
            drop(pass);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..default()
        });
        pass.set_pipeline(&self.pipelines.blitting);
        pass.set_bind_group(0, &self.pipelines.samples_bind_group, &[]);
        pass.set_bind_group(1, &self.pipelines.metadata_bind_group, &[]);
        pass.set_push_constants(ShaderStages::FRAGMENT, 0, &pc_buf);
        pass.draw(0 .. 6, 0 .. 1);
        drop(pass);

        let commands = encoder.finish();
        let submission = self.queue.submit([commands]);
        self.window.pre_present_notify();
        surface_texture.present();

        self.device.poll(PollType::WaitForSubmissionIndex(submission)).unwrap();

        let end = Instant::now();
        eprintln!("rendering took {:.02}ms", (end - now).as_secs_f64() * 1000.0);
    }
}

fn default<T: Default>() -> T {
    T::default()
}
