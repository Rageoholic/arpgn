// FILE SAFETY REQUIREMENTS
// * Keep an Arc to parent device
// * Do not destroy inner unless you are in drop
// * Do not destroy the return result of any function that returns inner
// * Do not create except in new

use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk::{PipelineLayout as RawPipelineLayout, PipelineLayoutCreateInfo};

use super::utils::associate_debug_name;
use super::Device;

#[derive(Debug)]
pub struct PipelineLayout {
    inner: RawPipelineLayout,
    parent: Arc<Device>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        //SAFETY: This is drop where we *can* do this
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_pipeline_layout(self.inner, None)
        }
    }
}

impl PipelineLayout {
    //SAFETY REQUIREMENTS: Valid ci
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &PipelineLayoutCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let inner =
            // SAFETY: valid ci. Preconditions of this unsafe function
            unsafe { device.as_inner_ref().create_pipeline_layout(ci, None) }?;
        associate_debug_name!(device, inner, debug_name);
        Ok(Self {
            inner,
            parent: device.clone(),
        })
    }

    pub(crate) fn get_inner(&self) -> RawPipelineLayout {
        self.inner
    }
}
