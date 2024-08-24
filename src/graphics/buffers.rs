use std::{marker::PhantomData, sync::Arc};

use ash::{prelude::VkResult, vk::BufferCreateInfo};
use bytemuck::{Pod, Zeroable};
use vk_mem::{Alloc, AllocationCreateInfo as BufferAllocationCreateInfo};

use super::Device;

#[derive(Debug)]
pub struct ManagedMappableBuffer<T> {
    parent: Arc<Device>,
    _phantom: PhantomData<[T]>,
    inner: ash::vk::Buffer,
    allocation: vk_mem::Allocation,
}

impl<T: Pod + Zeroable> ManagedMappableBuffer<T> {
    //SAFETY: ai must have memory be host visible
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &BufferCreateInfo,
        ai: &BufferAllocationCreateInfo,
    ) -> VkResult<Self> {
        let (inner, allocation) =
            //SAFETY: Valid ci, valid ai
            unsafe { device.get_allocator_ref().create_buffer(ci, ai) }?;

        Ok(Self {
            parent: device.clone(),
            inner,
            allocation,
            _phantom: PhantomData,
        })
    }
}

impl<T> Drop for ManagedMappableBuffer<T> {
    fn drop(&mut self) {
        //SAFETY: Tied together
        unsafe {
            self.parent
                .get_allocator_ref()
                .destroy_buffer(self.inner, &mut self.allocation)
        }
    }
}
