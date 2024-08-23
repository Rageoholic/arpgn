// FILE SAFETY INVARIANTS
// * We are dropped after the Entry passed to us in Instance::new. This is
//   assured by holding onto an Arc to it, since Arc won't drop it until the
//   last reference gets dropped.

// * We only create Instance in Instance::new. Do not construct an instance in
//   any other way

// * NEVER EVER EVER CALL `destroy_device` ON THE RETURN FROM `as_inner_ref`

// * Do not destroy inner except in drop()

use std::sync::Arc;

use ash::{
    prelude::VkResult,
    vk::{self, InstanceCreateInfo},
    Entry,
};

pub(super) struct Instance {
    inner: ash::Instance,
    parent: Arc<Entry>,
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
        entry: &Arc<Entry>,
        ci: &InstanceCreateInfo,
    ) -> VkResult<Self> {
        //SAFETY: valid ci. From function safety requirements
        unsafe { entry.create_instance(ci, None) }.map(|inner| Self {
            inner,
            parent: entry.clone(),
        })
    }

    pub(super) fn parent(&self) -> &Arc<Entry> {
        &self.parent
    }

    /// NEVER EVER CALL `destroy_instance` ON THE RETURN VALUE FROM THIS
    pub(super) fn as_inner_ref(&self) -> &ash::Instance {
        &self.inner
    }

    pub(super) fn get_physical_devices(
        &self,
    ) -> VkResult<Vec<vk::PhysicalDevice>> {
        //SAFETY: Should always be safe
        unsafe { self.inner.enumerate_physical_devices() }
    }
    //SAFETY REQUIREMENTS: phys_dev must be derived from self
    pub(super) unsafe fn get_relevant_physical_device_properties(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> PhysDevProps {
        let props =
        //SAFETY: phys_dev from self
            unsafe { self.inner.get_physical_device_properties(phys_dev) };
        let features =
            //SAFETY: phys_dev from self
            unsafe { self.inner.get_physical_device_features(phys_dev) };
        //SAFETY: phys_dev_from self
        let extensions = unsafe {
            self.inner.enumerate_device_extension_properties(phys_dev)
        }
        .unwrap();
        //SAFETY: phys_dev from self
        let queue_families = unsafe {
            self.inner
                .get_physical_device_queue_family_properties(phys_dev)
        };
        PhysDevProps {
            props,
            _features: features,
            queue_families,
            extensions,
        }
    }
}

pub struct PhysDevProps {
    pub props: vk::PhysicalDeviceProperties,
    pub _features: vk::PhysicalDeviceFeatures,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub extensions: Vec<vk::ExtensionProperties>,
}
