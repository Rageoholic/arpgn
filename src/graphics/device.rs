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
    ffi::CString,
    fmt::Debug,
    mem::ManuallyDrop,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::{
    prelude::VkResult,
    vk::{
        DebugUtilsObjectNameInfoEXT, DeviceCreateInfo, Fence, Handle,
        PhysicalDevice, Queue, Result as RawVkResult, SubmitInfo,
    },
};

use super::instance::Instance;

pub struct Device {
    inner: ash::Device,
    phys_dev: PhysicalDevice,
    parent: Arc<super::Instance>,
    queue_families: HashMap<u32, Vec<Mutex<Queue>>>,
    allocator: ManuallyDrop<vk_mem::Allocator>,
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("inner", &self.inner.handle().as_raw())
            .field("phys_dev", &self.phys_dev)
            .field("parent", &self.parent)
            .field("_queue_families", &self.queue_families)
            .finish()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //SAFETY: Last use of self.allocator
        unsafe { ManuallyDrop::drop(&mut self.allocator) };

        //SAFETY: Last use of device, all child objects have been destroyed
        //(other objects are responsible for that)

        unsafe { self.inner.destroy_device(None) };
    }
}

#[derive(Debug)]
pub enum QueueRetrievalError {
    NoSuchQueue,
}

impl Device {
    pub fn is_debug(&self) -> bool {
        self.debug_utils_device.is_some()
    }
    //SAFETY REQUIREMENTS: Valid ci and phys_dev derived from instance
    pub unsafe fn new(
        instance: &Arc<super::Instance>,
        phys_dev: PhysicalDevice,
        ci: &DeviceCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        //SAFETY: valid ci. phys_dev derived from instance. Preconditions of
        //this unsafe function
        let inner = unsafe {
            instance.as_inner_ref().create_device(phys_dev, ci, None)
        }?;

        //SAFETY: Must be safe if this is a valid ci
        let queue_cis = unsafe {
            slice::from_raw_parts(
                ci.p_queue_create_infos,
                ci.queue_create_info_count as usize,
            )
        };
        let mut queue_families = HashMap::with_capacity(queue_cis.len());

        for queue_ci in queue_cis {
            let family = queue_ci.queue_family_index;
            let family_queue_count = queue_ci.queue_count;
            let mut queue_family_queues =
                Vec::with_capacity(family_queue_count as usize);
            for i in 0..family_queue_count {
                queue_family_queues.push(Mutex::new(
                    //SAFETY: We know these queues exist cause we're
                    //pulling from the queue family info
                    unsafe { inner.get_device_queue(family, i) },
                ));
            }

            queue_families.insert(family, queue_family_queues);
        }
        let allocator_ci = vk_mem::AllocatorCreateInfo::new(
            instance.as_inner_ref(),
            &inner,
            phys_dev,
        );
        let allocator =
            //SAFETY: Valid ci
            ManuallyDrop::new(unsafe { vk_mem::Allocator::new(allocator_ci) }?);
        let debug_utils_device = instance.is_debug().then(|| {
            let debug_device = ash::ext::debug_utils::Device::new(
                instance.as_inner_ref(),
                &inner,
            );
            if let Some(dn) = debug_name {
                let debug_name = CString::new(dn).unwrap();
                let name_info = DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(inner.handle())
                    .object_name(&debug_name);
                unsafe { debug_device.set_debug_utils_object_name(&name_info) }
                    .unwrap();
            }

            debug_device
        });

        Ok(Self {
            inner,
            phys_dev,
            parent: instance.clone(),
            queue_families,
            allocator,
            debug_utils_device,
        })
    }

    pub unsafe fn get_queue(
        &self,
        family_index: u32,
        queue_index: u32,
    ) -> Result<MutexGuard<Queue>, QueueRetrievalError> {
        self.queue_families
            .get(&family_index)
            .and_then(|queues| queues.get(queue_index as usize))
            .map(|locked_queue| {
                locked_queue.lock().unwrap_or_else(|p| p.into_inner())
            })
            .ok_or(QueueRetrievalError::NoSuchQueue)
    }

    //SAFETY Requirements: 1) Valid submits, 2) fence comes from this device
    pub unsafe fn submit_command_buffers(
        &self,

        family: u32,
        queue_index: u32,
        submits: &[SubmitInfo],
        fence: Fence,
    ) -> QueueSubmitResult {
        let lock = self
            .queue_families
            .get(&family)
            .ok_or(QueueSubmitError::NoSuchQueue)?
            .get(queue_index as usize)
            .ok_or(QueueSubmitError::NoSuchQueue)?
            .lock()
            //We can just ignore poisoning. We hold this for so little time that it's irrelevant
            .unwrap_or_else(|p| p.into_inner());
        //SAFETY: 1) valid submits, 2) fence comes from this device, 3) Only one
        // thread can be submitting at a time. We ensure 1 and 2 via our own
        // preconditions and we ensure 3 by ensuring that this is the only place
        // where you can submit to a queue.
        unsafe { self.inner.queue_submit(*lock, submits, fence) }
            .map_err(QueueSubmitError::Vulkan)
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
    pub(super) fn get_allocator_ref(&self) -> &vk_mem::Allocator {
        &self.allocator
    }

    pub fn wait_idle(&self) -> VkResult<()> {
        //SAFETY: We basically know this is always safe
        unsafe { self.inner.device_wait_idle() }
    }

    pub fn associate_debug_name(
        &self,
        name_info: &DebugUtilsObjectNameInfoEXT,
    ) {
        unsafe {
            self.debug_utils_device.as_ref().map(|debug| {
                debug.set_debug_utils_object_name(name_info).unwrap()
            })
        };
    }
}

#[derive(thiserror::Error, Debug)]
pub enum QueueSubmitError {
    #[error("Raw vk error {0:?}")]
    Vulkan(RawVkResult),
    #[error("Requested submission to invalid queue")]
    NoSuchQueue,
}

pub type QueueSubmitResult = Result<(), QueueSubmitError>;
