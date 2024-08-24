// FILE SAFETY REQUIREMENTS
// * Only create Swapchain in Swapchain::new

// * Do not destroy inner except in the Drop implementation

// * We must hold an Arc to our parent surface and device

use std::{cmp::min, sync::Arc};

use ash::{prelude::VkResult, vk};

use super::{device::Device, Surface};

pub(super) struct Swapchain {
    swapchain_device: ash::khr::swapchain::Device,
    inner: vk::SwapchainKHR,
    //NOTE: Exist for RAII reasons. These will keep the device and surface open
    //until this drops
    parent_device: Arc<Device>,
    _parent_surface: Arc<Surface>,
    _format: vk::SurfaceFormatKHR,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
}

impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("inner", &self.inner)
            .field("parent_device", &self.parent_device)
            .field("_parent_surface", &self._parent_surface)
            .field("_format", &self._format)
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

impl Swapchain {
    //SAFETY REQUIREMENT: device and surface are from the same instance
    pub unsafe fn new(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        graphics_qfi: u32,
        present_qfi: u32,
    ) -> VkResult<Self> {
        //SAFETY: surface and device are created from the same instance
        let swap_info = unsafe {
            surface.get_compatible_swapchain_info(
                device.get_physical_device_handle(),
            )
        }?;
        let format = swap_info
            .formats
            .iter()
            .find(|format| {
                format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    && format.format == vk::Format::B8G8R8A8_SRGB
            })
            .copied()
            //NOTE: Crashing because somehow we got here with no available
            //formats is fine
            .unwrap_or(swap_info.formats[0]);
        let present_mode = swap_info
            .present_modes
            .iter()
            .copied()
            .find(|pm| *pm == vk::PresentModeKHR::MAILBOX)
            .or_else(|| {
                swap_info
                    .present_modes
                    .iter()
                    .copied()
                    .find(|pm| *pm == vk::PresentModeKHR::IMMEDIATE)
            })
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let win_size = surface.get_size();

        let swap_extent =
            if swap_info.capabilities.current_extent.width != u32::MAX {
                swap_info.capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: win_size.width.clamp(
                        swap_info.capabilities.min_image_extent.width,
                        swap_info.capabilities.max_image_extent.width,
                    ),
                    height: win_size.height.clamp(
                        swap_info.capabilities.min_image_extent.height,
                        swap_info.capabilities.max_image_extent.height,
                    ),
                }
            };

        let image_count = min(
            swap_info.capabilities.min_image_count + 1,
            if swap_info.capabilities.max_image_count != 0 {
                swap_info.capabilities.max_image_count
            } else {
                u32::MAX
            },
        );
        let mut indices = Vec::with_capacity(2);
        let sharing_mode = if graphics_qfi == present_qfi {
            vk::SharingMode::EXCLUSIVE
        } else {
            indices.push(graphics_qfi);
            indices.push(present_qfi);
            vk::SharingMode::CONCURRENT
        };

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.get_inner())
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(swap_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&indices)
            .pre_transform(swap_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain_device = ash::khr::swapchain::Device::new(
            device.parent().as_inner_ref(),
            device.as_inner_ref(),
        );

        //SAFETY: Valid ci. Known because we did it
        let inner = unsafe { swapchain_device.create_swapchain(&ci, None) }?;

        let swapchain_images =
            //SAFETY: Should always be good
            unsafe { swapchain_device.get_swapchain_images(inner) }
            .expect("Got a swapchain with no images?");

        let mut image_views = Vec::with_capacity(swapchain_images.len());

        for image in swapchain_images {
            let components = vk::ComponentMapping::default();
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let image_view_ci = vk::ImageViewCreateInfo::default()
                .components(components)
                .subresource_range(subresource_range)
                .format(format.format)
                .view_type(vk::ImageViewType::TYPE_2D)
                .image(image);
            //SAFETY: We know image_view_ci is valid
            let image_view = unsafe {
                device
                    .as_inner_ref()
                    .create_image_view(&image_view_ci, None)
            }
            .unwrap();

            image_views.push(image_view);
        }

        Ok(Swapchain {
            inner,
            swapchain_device,
            image_views,
            _format: format,
            parent_device: device.clone(),
            _parent_surface: surface.clone(),
            extent: swap_extent,
        })
    }
    pub fn default_viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0f32,
            y: 0 as f32,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
    pub fn default_scissor(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        //SAFETY: We made this swapchain in new, which makes it with the same
        //swapchain_device we're using to destroy it now
        unsafe {
            for iv in &self.image_views {
                self.parent_device
                    .as_inner_ref()
                    .destroy_image_view(*iv, None);
            }
            self.image_views.clear();
            self.swapchain_device.destroy_swapchain(self.inner, None);
        }
    }
}
