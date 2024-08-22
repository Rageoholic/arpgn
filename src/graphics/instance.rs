use std::rc::Rc;

use ash::{prelude::VkResult, vk::InstanceCreateInfo, Entry};

pub(super) struct Instance {
    inner: ash::Instance,
    parent: Rc<Entry>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        //SAFETY: Last thing we do with the instance
        unsafe { self.inner.destroy_instance(None) };
    }
}

impl Instance {
    //SAFETY REQUIREMENTS: Must provide a valid ci
    pub(super) unsafe fn new(
        entry: &Rc<Entry>,
        ci: &InstanceCreateInfo,
    ) -> VkResult<Self> {
        //SAFETY: valid ci. From function safety requirements
        unsafe { entry.create_instance(ci, None) }.map(|inner| Self {
            inner,
            parent: entry.clone(),
        })
    }

    pub(super) fn parent(&self) -> &Rc<Entry> {
        &self.parent
    }
    pub(super) fn as_inner(&self) -> &ash::Instance {
        &self.inner
    }
}
