#![allow(dead_code, unused_imports, unused_variables)]

pub mod rendering_engine;
pub mod util;

use std::{
    borrow::Cow, collections::{hash_map, HashMap}, fs::metadata, ops::Deref, path::PathBuf, sync::Arc, time::Instant
};

use clap::Parser;
use rand::Rng;
use wgpu::{
    BufferUsages, PollType, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::Window,
};

use crate::{rendering_engine::RenderingEngine, util::*};

#[derive(Debug, Parser)]
#[command(disable_help_flag = true)]
struct AppArgs {
    #[arg(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,

    #[arg(short, long, default_value_t = 512)]
    width: u32,
    
    #[arg(short = 'y', long)]
    overwrite_output: bool,
    
    output_image: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = AppArgs::parse();
    #[cfg(debug_assertions)]
    dbg!(&args);
    
    /* let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?; */

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
            state: None,
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
            height: args.width,
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
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: key_state,
                        repeat,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Escape) => {
                    event_loop.exit();
                }
                _ if !repeat && key_state == ElementState::Pressed => {
                    self.window.request_redraw();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                // self.window.request_redraw();
                self.render();
            }

            // just to hush logging
            WindowEvent::CursorMoved { .. }
            | WindowEvent::Focused(..)
            | WindowEvent::Moved(..)
            | WindowEvent::CursorEntered { .. }
            | WindowEvent::CursorLeft { .. }
            | WindowEvent::ModifiersChanged(..)
            | WindowEvent::MouseInput { .. }
            | WindowEvent::MouseWheel { .. } => {}
            _ => eprintln!("unknown window event {event:#?}"),
        }
    }
}

struct AppState {
    window: Arc<Window>,
    rendering_engine: RenderingEngine,
}

impl AppState {
    async fn new(args: AppArgs, window: Window) -> Self {
        let window = Arc::new(window);
        let surface_source = rendering_engine::SurfaceSource::Window(window.clone());
        let rendering_engine = RenderingEngine::new(surface_source, default(), default()).await;
        Self {
            window,
            rendering_engine,
        }
    }
    
    fn render(&self) {
        /*let now = Instant::now();
        let iters_per_invocation = 500u32;
        // FIXME: pass 64 bit time
        let now_secs = (now - self.epoch).as_secs_f32();

        let mut pc_buf = [0u8; size_of::<(u32, u32, f32, u32)>()];
        (&mut pc_buf[0..4]).copy_from_slice(&self.args.width.to_ne_bytes());
        (&mut pc_buf[4..8]).copy_from_slice(&self.args.height.to_ne_bytes());
        (&mut pc_buf[8..12]).copy_from_slice(&now_secs.to_ne_bytes());
        (&mut pc_buf[12..16]).copy_from_slice(&iters_per_invocation.to_ne_bytes());

        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("could not get surface's texture");
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..default()
            });

        let mut encoder = self.device.create_command_encoder(&default());

        for _ in 0..1 {
            let now = Instant::now();
            let now_secs = (now - self.epoch).as_secs_f32();
            (&mut pc_buf[8..12]).copy_from_slice(&now_secs.to_ne_bytes());

            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&self.pipelines.compute);
            pass.set_bind_group(0, &self.pipelines.samples_bind_group, &[]);
            pass.set_bind_group(1, &self.pipelines.metadata_bind_group, &[]);
            pass.set_push_constants(0, &pc_buf);
            pass.dispatch_workgroups(64 * 1000, 1, 1);
            drop(pass);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..default()
        });
        pass.set_pipeline(&self.pipelines.blitting);
        pass.set_bind_group(0, &self.pipelines.samples_bind_group, &[]);
        pass.set_bind_group(1, &self.pipelines.metadata_bind_group, &[]);
        pass.set_push_constants(ShaderStages::FRAGMENT, 0, &pc_buf);
        pass.draw(0..6, 0..1);
        drop(pass);

        let commands = encoder.finish();
        let submission = self.queue.submit([commands]);
        self.window.pre_present_notify();
        surface_texture.present();

        self.device
            .poll(PollType::WaitForSubmissionIndex(submission))
            .unwrap();

        let end = Instant::now();
        eprintln!(
            "rendering took {:.02}ms",
            (end - now).as_secs_f64() * 1000.0
        );*/
        
        let surface_handle = self.rendering_engine.surface_dest.get_handle();
        let surface_view = surface_handle.get_view();
        
        let mut encoder = self.rendering_engine.device.create_command_encoder(&default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.rendering_engine.pipelines.render_pipeline);
        pass.draw(0 .. 3, 0 .. 1);
        drop(pass);
        
        let commands = encoder.finish();
        let submission = self.rendering_engine.queue.submit([commands]);
        self.rendering_engine.surface_dest.pre_present();
        surface_handle.present();
        
        self.rendering_engine.device.poll(PollType::WaitForSubmissionIndex(submission)).unwrap();
    }
}
