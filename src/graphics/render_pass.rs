use std::sync::Arc;

use ash::{
    prelude::VkResult, vk::RenderPass as RawRenderPass,
    vk::RenderPassCreateInfo,
};

use super::{device::Device, utils::associate_debug_name};

#[derive(Debug)]
pub struct RenderPass {
    inner: RawRenderPass,
    parent: Arc<Device>,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        //SAFETY: inner came from parent
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_render_pass(self.inner, None)
        };
    }
}

impl RenderPass {
    // SAFETY REQUIREMENTS: valid ci
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &RenderPassCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let inner =
            //SAFETY: valid ci
            unsafe { device.as_inner_ref().create_render_pass(ci, None) }?;

        associate_debug_name!(device, inner, debug_name);

        Ok(Self {
            inner,
            parent: device.clone(),
        })
    }

    pub(crate) fn get_inner(&self) -> RawRenderPass {
        self.inner
    }
}
