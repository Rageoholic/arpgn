// FILE SAFETY REQUIREMENTS
// * Keep an Arc to parent device
// * Do not destroy inner unless you are in drop
// * Do not destroy the return result of any function that returns inner
// * Do not create except in new

use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use super::Device;
pub struct PipelineLayout {
    inner: vk::PipelineLayout,
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
        ci: &vk::PipelineLayoutCreateInfo,
    ) -> VkResult<Self> {
        let inner =
            // SAFETY: valid ci. Preconditions of this unsafe function
            unsafe { device.as_inner_ref().create_pipeline_layout(ci, None) }?;
        Ok(Self {
            inner,
            parent: device.clone(),
        })
    }
}
