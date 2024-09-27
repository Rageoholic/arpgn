#![allow(dead_code)]
use std::{marker::PhantomData, ptr::copy_nonoverlapping, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        self, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, CommandBufferBeginInfo,
        CommandBufferLevel, MemoryPropertyFlags, SharingMode,
    },
};
use bytemuck::{Pod, Zeroable};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo as BufferAllocationCreateInfo,
    MemoryUsage,
};

use crate::{
    graphics::{command_buffers::CommandBuffer, sync_objects::Fence},
    utils::debug_string,
};

use super::{
    command_buffers::CommandPool,
    utils::{associate_debug_name, FenceProducer},
    Device,
};

#[derive(Debug)]
pub struct MappableBuffer<T: Pod + Zeroable> {
    parent: Arc<Device>,
    _phantom: PhantomData<[T]>,
    inner: ash::vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
}

impl<T: Pod + Zeroable> MappableBuffer<T> {
    pub fn new(
        device: &Arc<Device>,
        size: u64,
        usage: BufferUsageFlags,
        memory_preference: MemoryUsage,
        concurrent_queue_families: Option<&[u32]>,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let ci = BufferCreateInfo::default()
            .sharing_mode(if concurrent_queue_families.is_some() {
                SharingMode::CONCURRENT
            } else {
                SharingMode::EXCLUSIVE
            })
            .queue_family_indices(concurrent_queue_families.unwrap_or(&[]))
            .usage(usage)
            .size(size);
        let ai = BufferAllocationCreateInfo {
            flags: (AllocationCreateFlags::MAPPED
                | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE),
            usage: memory_preference,
            required_flags: MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        unsafe { Self::from_raw_cis(device, &ci, &ai, debug_name) }
    }
    //SAFETY: ai must have memory be host visible
    unsafe fn from_raw_cis(
        device: &Arc<Device>,
        ci: &BufferCreateInfo,
        ai: &BufferAllocationCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let (inner, allocation) = unsafe { device.get_allocator_ref().create_buffer(ci, ai) }?;
        associate_debug_name!(device, inner, debug_name);

        Ok(Self {
            parent: device.clone(),
            inner,
            allocation_info: device.get_allocator_ref().get_allocation_info(&allocation),
            allocation,
            _phantom: PhantomData,
        })
    }

    pub(crate) fn get_inner(&self) -> ash::vk::Buffer {
        self.inner
    }

    pub(crate) fn upload_data(&mut self, data: &[T]) {
        let mapping = self.allocation_info.mapped_data;

        assert!(self.allocation_info.size as usize >= size_of_val(data));

        //SAFETY: We just checked that the allocation is big enough for this
        unsafe { copy_nonoverlapping(data.as_ptr(), mapping as *mut T, data.len()) };

        self.parent
            .get_allocator_ref()
            .flush_allocation(&self.allocation, 0, size_of_val(data) as u64)
            .unwrap();
    }
}

impl<T: Pod + Zeroable> Drop for MappableBuffer<T> {
    fn drop(&mut self) {
        //SAFETY: Tied together
        unsafe {
            self.parent
                .get_allocator_ref()
                .destroy_buffer(self.inner, &mut self.allocation)
        }
    }
}

#[derive(Debug)]
pub struct GpuBuffer<T: Pod> {
    parent: Arc<Device>,
    inner: Buffer,
    allocation: Allocation,
    len: u64,
    _phantom: PhantomData<T>,
}

impl<T: Pod> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.parent
                .get_allocator_ref()
                .destroy_buffer(self.inner, &mut self.allocation);
        };
    }
}

#[derive(Debug)]
struct ReturnStaging<T: Pod> {
    staging_buffer: MappableBuffer<T>,
    fence: Fence,
    cb: CommandBuffer,
}
impl<T: Pod> FenceProducer for ReturnStaging<T> {
    type Iter = std::iter::Once<vk::Fence>;

    fn get_fences(&self) -> Self::Iter {
        std::iter::once(self.fence.get_inner())
    }
}

enum Return<T: Pod> {
    ReturnStaging(ReturnStaging<T>),
    ReturnMappable,
}

enum ReturnIter {
    Empty(std::iter::Empty<vk::Fence>),
    Once(std::iter::Once<vk::Fence>),
}

impl Iterator for ReturnIter {
    type Item = vk::Fence;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ReturnIter::Empty(empty) => empty.next(),
            ReturnIter::Once(once) => once.next(),
        }
    }
}

impl<T: Pod> FenceProducer for Return<T> {
    type Iter = ReturnIter;

    fn get_fences(&self) -> Self::Iter {
        match self {
            Return::ReturnStaging(return_staging) => ReturnIter::Once(return_staging.get_fences()),
            Return::ReturnMappable => ReturnIter::Empty(std::iter::empty()),
        }
    }
}
impl<T: Pod> GpuBuffer<T> {
    pub fn get_inner(&self) -> Buffer {
        self.inner
    }

    pub fn from_slice(
        device: &Arc<Device>,
        data: &[T],
        usage: BufferUsageFlags,
        storage: MemoryUsage,
        concurrent_queue_families: Option<&[u32]>,
        transfer_pool: &Arc<CommandPool>,
        transfer_qfi: u32,
        debug_name: Option<String>,
    ) -> VkResult<(Self, impl FenceProducer)> {
        let ci = BufferCreateInfo::default()
            .size(size_of_val(data) as u64)
            .sharing_mode(if concurrent_queue_families.is_some() {
                SharingMode::CONCURRENT
            } else {
                {
                    SharingMode::EXCLUSIVE
                }
            })
            .queue_family_indices(concurrent_queue_families.unwrap_or(&[]))
            .usage(usage | BufferUsageFlags::TRANSFER_DST);
        let ai = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: storage,
            ..Default::default()
        };
        let allocator = device.get_allocator_ref();
        let (inner, mut allocation) = unsafe { allocator.create_buffer(&ci, &ai) }?;

        let allocation_properties = { allocator.get_allocation_info(&allocation) };
        let memory_properties = unsafe { allocator.get_memory_properties() };
        let memory_type_info =
            memory_properties.memory_types_as_slice()[allocation_properties.memory_type as usize];
        if memory_type_info
            .property_flags
            .contains(MemoryPropertyFlags::HOST_VISIBLE)
        {
            let mapping_ptr = unsafe { allocator.map_memory(&mut allocation) }.unwrap();
            unsafe { copy_nonoverlapping(data.as_ptr(), mapping_ptr as *mut T, data.len()) };
            unsafe { allocator.unmap_memory(&mut allocation) };
            Ok((
                Self {
                    parent: device.clone(),
                    inner,
                    allocation,
                    len: size_of_val(data) as u64,
                    _phantom: PhantomData,
                },
                Return::ReturnMappable,
            ))
        } else {
            let mut staging_buffer = MappableBuffer::new(
                device,
                size_of_val(data) as u64,
                BufferUsageFlags::TRANSFER_SRC,
                MemoryUsage::AutoPreferHost,
                None,
                debug_string!(
                    device.is_debug(),
                    "Transfer Queue for {}",
                    debug_name.clone().unwrap_or("unknown name".into())
                ),
            )?;

            staging_buffer.upload_data(data);

            let mut cb = transfer_pool.alloc_command_buffer(CommandBufferLevel::PRIMARY)?;

            let transfer_complete_fence = Fence::new(
                device,
                false,
                debug_string!(
                    device.is_debug(),
                    "Transfer Fence for {}",
                    debug_name.unwrap_or("unknown buffer".to_string())
                ),
            )?;

            cb.record_and_submit(
                transfer_qfi,
                0,
                &[],
                &[],
                &[],
                &CommandBufferBeginInfo::default(),
                transfer_complete_fence.get_inner(),
                |dev, cb| -> Result<(), vk::Result> {
                    unsafe {
                        dev.cmd_copy_buffer(
                            cb,
                            staging_buffer.get_inner(),
                            inner,
                            &[BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: size_of_val(data) as u64,
                            }],
                        );
                    }
                    Ok(())
                },
            )
            .unwrap();

            Ok((
                Self {
                    parent: device.clone(),
                    inner,
                    allocation,
                    len: size_of_val(data) as u64,
                    _phantom: PhantomData,
                },
                Return::ReturnStaging(ReturnStaging {
                    staging_buffer,
                    cb,
                    fence: transfer_complete_fence,
                }),
            ))
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }
}
