#![warn(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![allow(clippy::redundant_closure)]

use graphics::{Context, ShaderLoadingError};
use structopt::StructOpt;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
};

mod graphics;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(long, default_value = "None")]
    graphics_validation_level: graphics::ValidationLevel,
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
                    ..Default::default()
                },
            ) {
                Ok(gc) => gc,
                Err(graphics::ContextCreationError::ShaderLoading(errs)) => {
                    for err in errs {
                        match err {
                            ShaderLoadingError::FileLoad(
                                file_name,
                                file_error,
                            ) => {
                                println!(
                                    "{} not found: {}",
                                    file_name.to_string_lossy(),
                                    file_error
                                );
                            }
                            ShaderLoadingError::ShaderCompilation(
                                compile_error,
                            ) => {
                                print!("{}", compile_error);
                            }
                            ShaderLoadingError::MemoryExhaustion => {
                                print!(
                                    "Memory exhausted while loading shaders"
                                );
                            }
                        }
                    }
                    event_loop.exit();
                    return;
                }
                Err(e) => panic!(
                    "Encountered error creating graphics context {:?}",
                    e
                ),
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
                    WindowEvent::Resized(_) => graphics_context.resize(),
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
    let opts = Opt::from_args();
    let event_loop = EventLoop::builder().build().unwrap();

    let mut app = App::Uninitialized(opts);

    event_loop.run_app(&mut app).unwrap();
}
