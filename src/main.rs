use std::sync::Arc;

use winit::{
    dpi::LogicalSize,
    event::{Event, StartCause, WindowEvent},
    event_loop::EventLoopBuilder,
    window::WindowBuilder,
};

const DEFAULT_WINDOW_WIDTH: i32 = 1280;

const DEFAULT_WINDOW_HEIGHT: i32 = 720;

fn main() {
    env_logger::init();
    let event_loop = EventLoopBuilder::new().build().unwrap();

    //win_opt is here so I can guarantee drop is called on all platforms and
    //because I remember *someone* saying that you're supposed to wait until you
    //are in the event loop before asking for a graphics context
    let mut win_opt = None;

    event_loop
        .run(move |event, loop_target| match win_opt {
            None => match event {
                Event::NewEvents(StartCause::Init) => {
                    win_opt = {
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
                        Some(win)
                    }
                }
                Event::LoopExiting => {}
                _ => {
                    log::warn!(target: "winit_event_loop", "Unexpected event {:?}", event)
                }
            },
            Some(ref win) => match event {
                Event::WindowEvent { window_id, event } => match event {
                    WindowEvent::CloseRequested => {
                        //I dunno if this guard is necessary but I am paranoid
                        //so whatever
                        if win.id() == window_id {
                            win.set_visible(false);
                            win_opt = None;
                            loop_target.exit()
                        }
                    }
                    _ => {}
                },
                _ => {}
            },
        })
        .unwrap()
}
