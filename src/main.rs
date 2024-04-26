#![warn(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
use std::sync::Arc;

use structopt::StructOpt;

use winit::{
    dpi::LogicalSize,
    event::{Event, StartCause, WindowEvent},
    event_loop::EventLoopBuilder,
    window::WindowBuilder,
};

mod graphics;

const DEFAULT_WINDOW_WIDTH: i32 = 1280;

const DEFAULT_WINDOW_HEIGHT: i32 = 720;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(long)]
    graphics_validation_layers: bool,
}

fn main() {
    env_logger::init();
    let opts = Opt::from_args();
    let event_loop = EventLoopBuilder::new().build().unwrap();

    //win_opt is here so I can guarantee drop is called on all platforms and
    //because I remember *someone* saying that you're supposed to wait until you
    //are in the event loop before asking for a graphics context
    let mut output_opt = None;

    event_loop
        .run(move |event, loop_target| match output_opt {
            None => match event {
                Event::NewEvents(StartCause::Init) => {
                    output_opt = {
                        let win = WindowBuilder::new()
                            .with_visible(false)
                            .with_inner_size(LogicalSize::new(
                                DEFAULT_WINDOW_WIDTH,
                                DEFAULT_WINDOW_HEIGHT,
                            ))
                            .build(loop_target)
                            .unwrap();

                        let win = Arc::new(win);

                        //initialization complete. We are ready to draw
                        //TODO: Eventually anyways
                        win.set_visible(true);
                        let graphics_context = graphics::Context::new(
                            win.clone(),
                            graphics::ContextCreateOpts {
                                graphics_validation_layers: opts.graphics_validation_layers,
                            },
                        )
                        .unwrap();
                        Some((win, graphics_context))
                    }
                }
                Event::LoopExiting => {}
                _ => {}
            },

            Some((ref win, _)) => match event {
                Event::WindowEvent { window_id, event } => match event {
                    WindowEvent::CloseRequested if win.id() == window_id => {
                        win.set_visible(false);
                        output_opt = None;
                        loop_target.exit()
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    //render here
                }
                _ => {}
            },
        })
        .unwrap()
}
