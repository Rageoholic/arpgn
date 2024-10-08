#![warn(unsafe_op_in_unsafe_fn)]
#![allow(clippy::redundant_closure, clippy::too_many_arguments)]

use clap::Parser;
use graphics::Context;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
};

mod graphics;
mod utils;

#[derive(Debug, Parser)]
struct Opt {
    #[arg(long, default_value = "None")]
    graphics_validation_level: graphics::ValidationLevel,
    #[arg(long)]
    unified_graphics_transfer_queue: bool,
    #[arg(long)]
    debuggable_shaders: bool,
}

#[derive(Debug)]
enum App {
    Uninitialized(Opt),
    Active(Context),
    Destroyed,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let App::Uninitialized(opts) = self {
            event_loop.set_control_flow(ControlFlow::Poll);

            let graphics_context = match graphics::Context::new(
                event_loop,
                graphics::ContextCreateOpts {
                    graphics_validation_layers: opts.graphics_validation_level,
                    unified_transfer_graphics_queue: opts.unified_graphics_transfer_queue,
                    debuggable_shaders: opts.debuggable_shaders,
                    ..Default::default()
                },
            ) {
                Ok(gc) => gc,
                Err(err) => {
                    eprintln!("{}", err);
                    event_loop.exit();
                    return;
                }
            };
            *self = App::Active(graphics_context);
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let App::Active(ref mut graphics_context) = self {
            if graphics_context.win_id() == window_id {
                match event {
                    WindowEvent::CloseRequested => {
                        *self = App::Destroyed;
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        graphics_context.draw();
                    }
                    WindowEvent::Resized(intended_size) => graphics_context.resize(intended_size),

                    _ => {}
                }
            }
        }
    }
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let App::Active(ref mut graphics_context) = self {
            graphics_context.draw()
        }
    }
}

fn main() {
    pretty_env_logger::init();
    let opts = Opt::parse();
    log::info!("Arguments from cmdline: {:?}", opts);
    let event_loop = EventLoop::builder().build().unwrap();

    let mut app = App::Uninitialized(opts);

    event_loop.run_app(&mut app).unwrap();
}
