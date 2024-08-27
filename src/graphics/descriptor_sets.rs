// FILE SAFETY REQUIREMENTS:
// * Users must not destroy the pool or the layout handle.
// * Only create a DescriptionMap in its new function
// * Only create a DescriptionSetLayout in its new function
// * We can only destroy vk handles in drop
// * We hold an Arc to the device that created us

use std::{marker::PhantomData, rc::Rc, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        DescriptorPoolCreateInfo, DescriptorSetAllocateInfo,
        DescriptorSetLayoutCreateInfo,
    },
};

use super::{Device, PhantomUnsendUnsync};

#[derive(Debug)]
pub struct DescriptorSetLayout {
    parent: Arc<Device>,
    inner: ash::vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        //SAFETY: inner comes from parent
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_descriptor_set_layout(self.inner, None)
        };
    }
}

impl DescriptorSetLayout {
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &DescriptorSetLayoutCreateInfo,
    ) -> VkResult<Self> {
        Ok(Self {
            //SAFETY: Valid ci
            inner: unsafe {
                device.as_inner_ref().create_descriptor_set_layout(ci, None)
            }?,
            parent: device.clone(),
        })
    }

    pub(crate) fn as_inner(&self) -> ash::vk::DescriptorSetLayout {
        self.inner
    }
}

#[derive(Debug)]
pub struct DescriptorPool {
    inner: ash::vk::DescriptorPool,
    _phantom: PhantomUnsendUnsync,
    parent: Arc<Device>,
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        //SAFETY: inner comes from parent
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_descriptor_pool(self.inner, None)
        };
    }
}

#[derive(Debug)]
pub struct DescriptorSet {
    _parent: Rc<DescriptorPool>,
    inner: ash::vk::DescriptorSet,
}

impl DescriptorPool {
    //SAFETY REQUIREMENTS: Valid CI
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &DescriptorPoolCreateInfo,
    ) -> VkResult<Self> {
        Ok(Self {
            //SAFETY: Valid ci
            inner: unsafe {
                device.as_inner_ref().create_descriptor_pool(ci, None)
            }?,
            parent: device.clone(),
            _phantom: PhantomData,
        })
    }
    pub fn as_inner(&self) -> ash::vk::DescriptorPool {
        self.inner
    }
}

impl DescriptorSet {
    pub unsafe fn alloc(
        pool: &Rc<DescriptorPool>,
        ai: &DescriptorSetAllocateInfo,
    ) -> VkResult<Vec<Self>> {
        //SAFETY: valid ai. ai.descriptor_pool == pool.inner
        unsafe { pool.parent.as_inner_ref().allocate_descriptor_sets(ai) }.map(
            |raw_handles| {
                raw_handles
                    .into_iter()
                    .map(|inner| Self {
                        inner,
                        _parent: pool.clone(),
                    })
                    .collect()
            },
        )
    }

    pub(crate) fn get_inner(&self) -> ash::vk::DescriptorSet {
        self.inner
    }
}
