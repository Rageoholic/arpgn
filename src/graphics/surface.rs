// FILE SAFETY REQUIREMENTS
// * Only create Surface in Surface::new

// * Do not destroy inner except in the Drop implementation

// * We must hold an Arc to our parent window and instance

use std::sync::Arc;

use ash::{
    prelude::VkResult,
    vk::{
        Handle, PhysicalDevice, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR,
    },
};
use winit::{
    dpi::PhysicalSize,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use super::Instance;

pub(super) struct Surface {
    surface: SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,
    //These are here to ensure these are dropped *after* the surface
    _parent_instance: Arc<Instance>,
    parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        //SAFETY: We made this surface with this surface instance. We only make
        //Surface in new which upholds this
        unsafe { self.surface_instance.destroy_surface(self.surface, None) };
    }
}
impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("surface", &self.surface.as_raw())
            .field("_parent_instance", &self._parent_instance)
            .field("parent_window", &self.parent_window)
            .finish_non_exhaustive()
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

        let surface_instance =
            ash::khr::surface::Instance::new(instance.parent(), instance.as_inner_ref());

        Ok(Self {
            surface,
            surface_instance,
            _parent_instance: instance.clone(),
            parent_window: win.clone(),
        })
    }
    //SAFETY REQUIREMENTS: phys_dev must be derived from same instance as
    //surface, queue_family_index must be in bounds
    pub unsafe fn does_queue_support_presentation(
        &self,
        phys_dev: PhysicalDevice,
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

    pub unsafe fn get_compatible_swapchain_info(
        &self,
        phys_dev: PhysicalDevice,
    ) -> VkResult<SwapchainInfo> {
        //SAFETY: phys_dev is derived from same parent instance as surface
        let capabilities = unsafe {
            self.surface_instance
                .get_physical_device_surface_capabilities(phys_dev, self.surface)
        }?;
        //SAFETY: phys_dev is derived from same parent instance as surface
        let formats = unsafe {
            self.surface_instance
                .get_physical_device_surface_formats(phys_dev, self.surface)
        }?;

        //SAFETY: phys_dev is derived from same parent instance as surface
        let present_modes = unsafe {
            self.surface_instance
                .get_physical_device_surface_present_modes(phys_dev, self.surface)?
        };

        Ok(SwapchainInfo {
            capabilities,
            formats,
            present_modes,
        })
    }
    pub fn get_size(&self) -> PhysicalSize<u32> {
        self.parent_window.inner_size()
    }
    pub fn get_inner(&self) -> SurfaceKHR {
        self.surface
    }
}

pub struct SwapchainInfo {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub present_modes: Vec<PresentModeKHR>,
    pub formats: Vec<SurfaceFormatKHR>,
}
