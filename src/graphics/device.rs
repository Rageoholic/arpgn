// FILE SAFETY INVARIANTS
// * We are dropped after the Instance passed to us in Device::new. This is
//   assured by holding onto an Arc to it, since Arc won't drop it until the
//   last reference gets dropped.

// * We only create Device in Device::new. Do not construct an instance in
//   any other way

// * Do not destroy the inner until drop()

// * NEVER EVER EVER CALL `destroy_device` ON THE RETURN FROM `as_inner_ref`

use core::slice;
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, RwLock},
};

use ash::{
    prelude::VkResult,
    vk::{
        DeviceCreateInfo, Fence, Handle, PhysicalDevice, Queue,
        Result as RawVkResult, SubmitInfo,
    },
};

use super::instance::Instance;

pub struct Device {
    inner: ash::Device,
    phys_dev: PhysicalDevice,
    parent: Arc<super::Instance>,
    _queue_families: HashMap<u32, Vec<RwLock<Queue>>>,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("inner", &self.inner.handle().as_raw())
            .field("phys_dev", &self.phys_dev)
            .field("parent", &self.parent)
            .field("_queue_families", &self._queue_families)
            .finish()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //SAFETY: Last use of device, all child objects have been destroyed
        //(other objects are responsible for that)
        unsafe { self.inner.destroy_device(None) };
    }
}

impl Device {
    //SAFETY REQUIREMENTS: Valid ci and phys_dev derived from instance
    pub unsafe fn new(
        instance: &Arc<super::Instance>,
        phys_dev: PhysicalDevice,
        ci: &DeviceCreateInfo,
    ) -> VkResult<Self> {
        //SAFETY: valid ci. phys_dev derived from instance. Preconditions of
        //this unsafe function
        unsafe { instance.as_inner_ref().create_device(phys_dev, ci, None) }
            .map(|inner| {
                //SAFETY: Must be safe if this is a valid ci
                let queue_cis = unsafe {
                    slice::from_raw_parts(
                        ci.p_queue_create_infos,
                        ci.queue_create_info_count as usize,
                    )
                };
                let mut queue_families =
                    HashMap::with_capacity(queue_cis.len());

                for queue_ci in queue_cis {
                    let family = queue_ci.queue_family_index;
                    let family_queue_count = queue_ci.queue_count;
                    let mut queue_family_queues =
                        Vec::with_capacity(family_queue_count as usize);
                    for i in 0..family_queue_count {
                        queue_family_queues.push(RwLock::new(
                            //SAFETY: We know these queues exist cause we're
                            //pulling from the queue family info
                            unsafe { inner.get_device_queue(family, i) },
                        ));
                    }

                    queue_families.insert(family, queue_family_queues);
                }

                Self {
                    inner,
                    phys_dev,
                    parent: instance.clone(),
                    _queue_families: queue_families,
                }
            })
    }

    //SAFETY Requirements: 1) Valid submits, 2) fence comes from this device
    unsafe fn _submit_command_buffers(
        &self,

        family: u32,
        queue_index: u32,
        submits: &[SubmitInfo],
        fence: Fence,
    ) -> _QueueSubmitResult {
        let lock = self
            ._queue_families
            .get(&family)
            .ok_or(_QueueSubmitError::NoSuchQueue)?
            .get(queue_index as usize)
            .ok_or(_QueueSubmitError::NoSuchQueue)?
            .write()
            //We can just ignore poisoning. We hold this for so little time that it's irrelevant
            .unwrap_or_else(|p| p.into_inner());
        //SAFETY: 1) valid submits, 2) fence comes from this device, 3) Only one
        // thread can be submitting at a time. We ensure 1 and 2 via our own
        // preconditions and we ensure 3 by ensuring that this is the only place
        // where you can submit to a queue.
        unsafe { self.inner.queue_submit(*lock, submits, fence) }
            .map_err(_QueueSubmitError::Vulkan)
    }
    pub(super) fn get_physical_device_handle(&self) -> PhysicalDevice {
        self.phys_dev
    }

    pub(super) fn parent(&self) -> &Arc<Instance> {
        &self.parent
    }

    /// NEVER EVER CALL `destroy_device` ON THE RETURN VALUE FROM THIS
    pub(super) fn as_inner_ref(&self) -> &ash::Device {
        &self.inner
    }
}
pub enum _QueueSubmitError {
    Vulkan(RawVkResult),
    NoSuchQueue,
}

pub type _QueueSubmitResult = Result<(), _QueueSubmitError>;
