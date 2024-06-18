#![warn(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
use std::sync::Arc;

use graphics::Context;
use structopt::StructOpt;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

mod graphics;

const DEFAULT_WINDOW_WIDTH: i32 = 1280;

const DEFAULT_WINDOW_HEIGHT: i32 = 720;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(long, default_value = "None")]
    graphics_validation_level: graphics::ValidationLevel,
}

#[derive(Debug)]
enum App {
    Uninitialized(Opt),
    Active(Arc<Window>, Context),
    Destroyed,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let App::Uninitialized(opts) = self {
            event_loop.set_control_flow(ControlFlow::Poll);
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_visible(false)
                            .with_inner_size(LogicalSize {
                                width: DEFAULT_WINDOW_WIDTH,
                                height: DEFAULT_WINDOW_HEIGHT,
                            }),
                    )
                    .unwrap(),
            );
            let graphics_context = match graphics::Context::new(
                window.clone(),
                graphics::ContextCreateOpts {
                    graphics_validation_layers: opts.graphics_validation_level,
                },
            ) {
                Ok(gc) => gc,
                Err(e) => panic!("Encountered error creating graphics context {:?}", e),
            };
            window.set_visible(true);
            *self = App::Active(window, graphics_context);
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let App::Active(ref win, ref mut graphcs_context) = self {
            if win.id() == window_id {
                match event {
                    WindowEvent::CloseRequested => {
                        *self = App::Destroyed;
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        graphcs_context.draw();
                    }
                    WindowEvent::Resized(_) => graphcs_context.resize(),
                    _ => {}
                }
            }
        }
    }
}

fn main() {
    pretty_env_logger::init();
    let opts = Opt::from_args();
    let event_loop = EventLoop::builder().build().unwrap();

    let mut app = App::Uninitialized(opts);

    event_loop.run_app(&mut app).unwrap();
}
