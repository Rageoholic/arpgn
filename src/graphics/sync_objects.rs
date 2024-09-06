use std::{ffi::CString, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{DebugUtilsObjectNameInfoEXT, FenceCreateFlags, FenceCreateInfo, SemaphoreCreateInfo},
};

use crate::graphics::utils::associate_debug_name;

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
    pub fn new(device: &Arc<Device>, debug_name: Option<String>) -> VkResult<Self> {
        let ci = SemaphoreCreateInfo::default();
        let inner = unsafe { device.as_inner_ref().create_semaphore(&ci, None)? };

        associate_debug_name!(device, inner, debug_name);
        Ok(Self {
            inner,
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
    pub fn new(device: &Arc<Device>, signaled: bool, debug_name: Option<String>) -> VkResult<Self> {
        let ci = FenceCreateInfo::default().flags(if signaled {
            FenceCreateFlags::SIGNALED
        } else {
            FenceCreateFlags::empty()
        });
        let inner = unsafe { device.as_inner_ref().create_fence(&ci, None)? };
        if device.is_debug() && debug_name.is_some() {
            let debug_name = CString::new(debug_name.unwrap()).unwrap();
            let name_info = DebugUtilsObjectNameInfoEXT::default()
                .object_handle(inner)
                .object_name(&debug_name);
            device.associate_debug_name(&name_info);
        }
        Ok(Self {
            inner,
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
            self.parent
                .as_inner_ref()
                .wait_for_fences(fences, false, u64::MAX)?;
            self.parent.as_inner_ref().reset_fences(fences)
        }
    }
}
