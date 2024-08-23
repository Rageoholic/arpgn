// FILE SAFETY REQUIREMENTS
// * Keep an Arc to parent device
// * Do not destroy inner unless you are in drop
// * Do not destroy the return result of any function that returns inner
// * Do not create except in new

use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use super::device::Device;

pub struct DescriptorSetLayout {
    inner: vk::DescriptorSetLayout,
    parent: Arc<super::Device>,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        //SAFETY: We are in drop and will not use inner after this
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_descriptor_set_layout(self.inner, None)
        };
    }
}

impl DescriptorSetLayout {
    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.inner
    }

    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &vk::DescriptorSetLayoutCreateInfo,
    ) -> VkResult<Self> {
        //SAFETY: Valid ci
        let inner = unsafe {
            device.as_inner_ref().create_descriptor_set_layout(ci, None)
        }?;
        Ok(Self {
            inner,
            parent: device.clone(),
        })
    }
}
