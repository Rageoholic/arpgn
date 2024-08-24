use std::{marker::PhantomData, rc::Rc, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        CommandBufferAllocateInfo, CommandBufferLevel, CommandPoolCreateInfo,
    },
};

use super::Device;

#[derive(Debug)]
pub struct CommandBuffer {
    _inner: ash::vk::CommandBuffer,
    _parent: Rc<CommandPool>,
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
    ) -> VkResult<Self> {
        let inner =
            //SAFETY: valid ci from caller
            unsafe { device.as_inner_ref().create_command_pool(ci, None) }?;
        Ok(CommandPool {
            inner,
            parent: device.clone(),
            _phantom: PhantomData,
        })
    }

    pub fn alloc_command_buffers(
        self: Rc<Self>,
        count: u32,
        level: CommandBufferLevel,
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
        for raw_cb in raw_cbs {
            cbs.push(CommandBuffer {
                _inner: raw_cb,
                _parent: self.clone(),
            })
        }

        Ok(cbs)
    }
}
