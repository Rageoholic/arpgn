use std::sync::Arc;

use ash::{prelude::VkResult, vk};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use super::Instance;

pub(super) struct Surface {
    surface: vk::SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,
    //These are here to ensure these are dropped *after* the surface
    _parent_instance: Arc<Instance>,
    _parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        //SAFETY: We made this surface with this surface instance. We only make
        //Surface in new which upholds this
        unsafe { self.surface_instance.destroy_surface(self.surface, None) };
    }
}

impl Surface {
    pub fn new(instance: &Arc<Instance>, win: &Arc<Window>) -> VkResult<Self> {
        //SAFETY: win can make a valid RawDisplayHandle for ash_window to
        //consume. Winit basically assures us this is the case.
        let surface = unsafe {
            ash_window::create_surface(
                instance.parent(),
                instance.as_inner_ref(),
                win.display_handle().unwrap().as_raw(),
                win.window_handle().unwrap().as_raw(),
                None,
            )
        }?;

        let surface_instance = ash::khr::surface::Instance::new(
            instance.parent(),
            instance.as_inner_ref(),
        );

        Ok(Self {
            surface,
            surface_instance,
            _parent_instance: instance.clone(),
            _parent_window: win.clone(),
        })
    }
    //SAFETY REQUIREMENTS: phys_dev must be derived from same instance as
    //surface, queue_family_index must be in bounds
    pub unsafe fn does_queue_support_presentation(
        &self,
        phys_dev: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> bool {
        //SAFETY: phys_dev from same instance, qfi in bounds. This function's
        //unsafe preconditions
        unsafe {
            self.surface_instance.get_physical_device_surface_support(
                phys_dev,
                queue_family_index,
                self.surface,
            )
        }
        .unwrap()
    }
}
