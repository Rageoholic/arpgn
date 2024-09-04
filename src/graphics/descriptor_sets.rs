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

use super::{utils::associate_debug_name, Device, PhantomUnsendUnsync};

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
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        //SAFETY: Valid ci
        let descriptor_set_layout = unsafe {
            device.as_inner_ref().create_descriptor_set_layout(ci, None)
        }?;
        associate_debug_name!(device, descriptor_set_layout, debug_name);
        Ok(Self {
            inner: descriptor_set_layout,
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
        debug_name: Option<String>,
    ) -> VkResult<Self> {
        let descriptor_pool =
            unsafe { device.as_inner_ref().create_descriptor_pool(ci, None) }?;
        associate_debug_name!(device, descriptor_pool, debug_name);
        Ok(Self {
            //SAFETY: Valid ci
            inner: descriptor_pool,
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
        mut f: Option<
            impl FnMut(usize, ash::vk::DescriptorSet) -> Option<String>,
        >,
    ) -> VkResult<Vec<Self>> {
        //SAFETY: valid ai. ai.descriptor_pool == pool.inner
        unsafe { pool.parent.as_inner_ref().allocate_descriptor_sets(ai) }.map(
            |raw_handles| {
                raw_handles
                    .into_iter()
                    .enumerate()
                    .map(|(i, inner)| {
                        if let Some(ref mut f) = f {
                            let debug_name = f(i, inner);
                            associate_debug_name!(
                                pool.parent,
                                inner,
                                debug_name
                            );
                        }
                        Self {
                            inner,
                            _parent: pool.clone(),
                        }
                    })
                    .collect()
            },
        )
    }

    pub(crate) fn get_inner(&self) -> ash::vk::DescriptorSet {
        self.inner
    }
}
