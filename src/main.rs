#![allow(dead_code, unused_imports, unused_variables)]

use std::{borrow::Cow, ops::Deref, sync::Arc, time::Instant};

use clap::Parser;
use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{Key, NamedKey}, window::Window};

#[derive(Debug, Parser)]
#[command(disable_help_flag = true)]
struct AppArgs {
    #[arg(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,

    #[arg(short, long, default_value_t = 960)]
    width: u32,

    #[arg(short, long, default_value_t = 640)]
    height: u32,
}

fn main() -> anyhow::Result<()> {
    let args = AppArgs::parse();
    #[cfg(debug_assertions)]
    dbg!(&args);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
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
        window_attrs.inner_size = Some(PhysicalSize {
            width: args.width,
            height: args.height,
        }.into());
        let window = event_loop.create_window(window_attrs).unwrap();
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
            WindowEvent::KeyboardInput { event, .. } => match event {
                KeyEvent { logical_key: Key::Named(NamedKey::Escape), .. } => {
                    event_loop.exit();
                },
                _ => {},
            },
            WindowEvent::RedrawRequested => {
                self.window.request_redraw();
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
    // chaos_game: wgpu::ComputePipeline,
    // rasterizer: wgpu::ComputePipeline,
    blitting: wgpu::RenderPipeline,
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quad.wgsl"))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
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
            blitting,
        }
    }

    fn render(&self) {
        let surface_texture = self.surface.get_current_texture().expect("could not get surface's texture");
        let texture_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.surface_format.add_srgb_suffix()),
            ..default()
        });

        let mut encoder = self.device.create_command_encoder(&default());
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
        pass.draw(0 .. 6, 0 .. 1);
        drop(pass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();
    }
}

fn default<T: Default>() -> T {
    T::default()
}