use std::{marker::PhantomData, ptr::copy_nonoverlapping, sync::Arc};

use ash::{prelude::VkResult, vk::BufferCreateInfo};
use bytemuck::{Pod, Zeroable};
use vk_mem::{Alloc, AllocationCreateInfo as BufferAllocationCreateInfo};

use super::{utils::associate_debug_name, Device};

#[derive(Debug)]
pub struct ManagedMappableBuffer<T> {
    parent: Arc<Device>,
    _phantom: PhantomData<[T]>,
    inner: ash::vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
}

impl<T: Pod + Zeroable> ManagedMappableBuffer<T> {
    //SAFETY: ai must have memory be host visible
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &BufferCreateInfo,
        ai: &BufferAllocationCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let (inner, allocation) =
            //SAFETY: Valid ci, valid ai
            unsafe { device.get_allocator_ref().create_buffer(ci, ai) }?;
        associate_debug_name!(device, inner, debug_name);

        Ok(Self {
            parent: device.clone(),
            inner,
            allocation_info: device
                .get_allocator_ref()
                .get_allocation_info(&allocation),
            allocation,
            _phantom: PhantomData,
        })
    }

    pub(crate) fn get_inner(&self) -> ash::vk::Buffer {
        self.inner
    }

    pub(crate) fn upload_data(&self, data: &[T]) {
        let mapping = self.allocation_info.mapped_data;

        assert!(self.allocation_info.size as usize >= size_of_val(data));

        //SAFETY: We just checked that the allocation is big enough for this
        unsafe {
            copy_nonoverlapping(data.as_ptr(), mapping as *mut T, data.len())
        };

        self.parent
            .get_allocator_ref()
            .flush_allocation(&self.allocation, 0, size_of_val(data) as u64)
            .unwrap();
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
