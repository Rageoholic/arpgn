use std::sync::Arc;

use ash::{
    prelude::VkResult,
    vk::{FenceCreateFlags, FenceCreateInfo, SemaphoreCreateInfo},
};

use super::Device;
#[derive(Debug)]
pub struct Semaphore {
    inner: ash::vk::Semaphore,
    parent: Arc<Device>,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        //SAFETY: Tied together
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_semaphore(self.inner, None)
        };
    }
}

impl Semaphore {
    pub fn new(device: &Arc<Device>) -> VkResult<Self> {
        let ci = SemaphoreCreateInfo::default();
        Ok(Self {
            //SAFETY: Valid ci
            inner: unsafe {
                device.as_inner_ref().create_semaphore(&ci, None)?
            },
            parent: device.clone(),
        })
    }
    pub fn get_inner(&self) -> ash::vk::Semaphore {
        self.inner
    }
}

#[derive(Debug)]
pub struct Fence {
    inner: ash::vk::Fence,
    parent: Arc<Device>,
}

impl Drop for Fence {
    fn drop(&mut self) {
        //SAFETY: Tied together
        unsafe { self.parent.as_inner_ref().destroy_fence(self.inner, None) };
    }
}

impl Fence {
    pub fn new(device: &Arc<Device>) -> VkResult<Self> {
        let ci = FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED);
        Ok(Self {
            //SAFETY: Valid ci
            inner: unsafe { device.as_inner_ref().create_fence(&ci, None)? },
            parent: device.clone(),
        })
    }
    pub fn get_inner(&self) -> ash::vk::Fence {
        self.inner
    }

    pub fn wait_and_reset(&mut self) -> VkResult<()> {
        let fences = &[self.inner];
        //SAFETY: Fence is known to come from device
        unsafe {
            self.parent.as_inner_ref().wait_for_fences(
                fences,
                false,
                u64::MAX,
            )?;
            self.parent.as_inner_ref().reset_fences(fences)
        }
    }
}
