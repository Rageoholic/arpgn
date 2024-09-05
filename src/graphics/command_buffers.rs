use std::{marker::PhantomData, rc::Rc, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandPoolCreateInfo, PipelineStageFlags, SubmitInfo,
    },
};

use super::{device::QueueSubmitError, utils::associate_debug_name, Device};

#[derive(Debug)]
pub struct CommandBuffer {
    inner: ash::vk::CommandBuffer,
    parent: Rc<CommandPool>,
}

#[derive(thiserror::Error, Debug)]
pub enum RecordAndSubmitError<T> {
    #[error("Error beginning/ending recording")]
    InternalRecording(ash::vk::Result),
    #[error("Error recording command buffer {0:?}")]
    UserRecording(#[from] T),
    #[error("Error submitting queue {0}")]
    QueueSubmitError(QueueSubmitError),
}

impl CommandBuffer {
    #[allow(clippy::too_many_arguments)]
    pub fn record_and_submit<T>(
        &mut self,
        queue_family: u32,
        queue_index: u32,
        wait_semaphores: &[ash::vk::Semaphore],
        wait_dst_stage_masks: &[PipelineStageFlags],
        signal_semaphores: &[ash::vk::Semaphore],
        begin_info: &CommandBufferBeginInfo,
        signal_fence: ash::vk::Fence,
        f: impl FnOnce(&ash::Device, ash::vk::CommandBuffer) -> Result<(), T>,
    ) -> Result<(), RecordAndSubmitError<T>> {
        let cb = self.inner;

        let device = self.parent.parent.as_inner_ref();
        //SAFETY: We know device and cb are related
        unsafe {
            device
                .begin_command_buffer(cb, begin_info)
                .map_err(|e| RecordAndSubmitError::InternalRecording(e))?;
            f(device, cb)?;
            device
                .end_command_buffer(cb)
                .map_err(|e| RecordAndSubmitError::InternalRecording(e))?;
            let cbs = &[cb];
            self.parent
                .parent
                .submit_command_buffers(
                    queue_family,
                    queue_index,
                    &[SubmitInfo::default()
                        .command_buffers(cbs)
                        .wait_semaphores(wait_semaphores)
                        .signal_semaphores(signal_semaphores)
                        .wait_dst_stage_mask(wait_dst_stage_masks)],
                    signal_fence,
                )
                .map_err(|e| RecordAndSubmitError::QueueSubmitError(e))?;
        }
        Ok(())
    }
}

type PhantomUnsendUnsync = PhantomData<*mut ()>;

#[derive(Debug)]
pub struct CommandPool {
    inner: ash::vk::CommandPool,
    parent: Arc<Device>,
    _phantom: PhantomUnsendUnsync,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        //SAFETY: inner derived from parent
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_command_pool(self.inner, None)
        };
    }
}

impl CommandPool {
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &CommandPoolCreateInfo,
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let inner =
            //SAFETY: valid ci from caller
            unsafe { device.as_inner_ref().create_command_pool(ci, None) }?;
        associate_debug_name!(device, inner, debug_name);
        Ok(CommandPool {
            inner,
            parent: device.clone(),
            _phantom: PhantomData,
        })
    }

    pub fn alloc_command_buffers(
        self: &Rc<Self>,
        count: u32,
        level: CommandBufferLevel,
        mut opt_f: Option<
            impl FnMut(usize, ash::vk::CommandBuffer) -> Option<String>,
        >,
    ) -> VkResult<Vec<CommandBuffer>> {
        let mut cbs = Vec::with_capacity(count as usize);
        let ai = CommandBufferAllocateInfo::default()
            .command_pool(self.inner)
            .command_buffer_count(count)
            .level(level);

        //SAFETY: Because CommandPool is !Send and !Sync we can be sure we're
        //the only ones accessing it. In addition, thanks to us being the only
        //vulkan struct passed we know everything comes from the same parent
        //device
        let raw_cbs = unsafe {
            self.parent.as_inner_ref().allocate_command_buffers(&ai)
        }?;
        for (i, raw_cb) in raw_cbs.into_iter().enumerate() {
            if let Some(ref mut f) = opt_f {
                if self.parent.is_debug() {
                    let debug_name = f(i, raw_cb);
                    associate_debug_name!(self.parent, raw_cb, debug_name);
                }
            }
            cbs.push(CommandBuffer {
                inner: raw_cb,
                parent: self.clone(),
            })
        }

        Ok(cbs)
    }

    pub(crate) fn alloc_command_buffer(
        self: &Rc<Self>,
        level: CommandBufferLevel,
    ) -> VkResult<CommandBuffer> {
        let ai = CommandBufferAllocateInfo::default()
            .command_pool(self.inner)
            .command_buffer_count(1)
            .level(level);
        //SAFETY: Device and inner are from same place
        let raw_cb = unsafe {
            self.parent.as_inner_ref().allocate_command_buffers(&ai)
        }?
        .pop()
        .unwrap();
        Ok(CommandBuffer {
            inner: raw_cb,
            parent: self.clone(),
        })
    }
}
