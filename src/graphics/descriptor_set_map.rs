// FILE SAFETY REQUIREMENTS:
// * Users must not destroy the pool or the layout handle.
// * Only create a DescriptionMap in its new function
// * Only create a DescriptionSetLayout in its new function
// * We can only destroy vk handles in drop
// * We hold an Arc to the device that created us

use std::{collections::HashMap, hash::Hash, marker::PhantomData, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
        DescriptorSetAllocateInfo, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
};

use super::{Device, PhantomUnsendUnsync};

#[derive(Debug)]
pub struct DescriptorSetMap<Key>
where
    Key: Hash + Eq,
{
    _inner_pool: DescriptorPool,
    _parent: Arc<Device>,
    _sets: HashMap<Key, Vec<DescriptorSet>>,
    layout: DescriptorSetLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorRequest {
    pub count: u32,
    pub binding: u32,
    pub ty: DescriptorType,
}

impl<Key: Hash + Eq> DescriptorSetMap<Key> {
    //SAFETY: All layouts in descriptor_requests must be from same device as
    //given
    pub fn new(
        device: &Arc<Device>,
        descriptor_requests: HashMap<Key, DescriptorRequest>,
    ) -> VkResult<Self> {
        let mut bindings = Vec::with_capacity(descriptor_requests.len());
        for request in descriptor_requests.values() {
            bindings.push(
                DescriptorSetLayoutBinding::default()
                    .binding(request.binding)
                    .descriptor_type(request.ty)
                    .descriptor_count(request.count),
            )
        }
        let descriptor_set_layout_ci =
            DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = DescriptorSetLayout {
            //SAFETY: Valid ci
            inner: unsafe {
                device.as_inner_ref().create_descriptor_set_layout(
                    &descriptor_set_layout_ci,
                    None,
                )
            }?,
            parent: device.clone(),
        };
        let mut descriptor_pool_sizes =
            Vec::with_capacity(descriptor_requests.len());

        let max_sets =
            descriptor_requests.iter().fold(0, |cum, (_, request)| {
                descriptor_pool_sizes.push(DescriptorPoolSize {
                    ty: request.ty,
                    descriptor_count: request.count,
                });
                cum + request.count
            });

        let ci = DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&descriptor_pool_sizes);

        let pool = DescriptorPool {
            //SAFETY: valid CI
            inner: unsafe {
                device.as_inner_ref().create_descriptor_pool(&ci, None)
            }?,
            _phantom: PhantomData,
            parent: device.clone(),
        };
        let mut sets = HashMap::with_capacity(descriptor_requests.len());

        for (k, request) in descriptor_requests {
            let duped_layouts = vec![layout.inner; request.count as usize];
            let ai = DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool.inner)
                .set_layouts(&duped_layouts)
                .to_owned();
            let bucket =
                //SAFETY: We know this is good because we created the descriptor_layout
                unsafe { device.as_inner_ref().allocate_descriptor_sets(&ai) }.expect("They basically guarantee that if we never free and allocate exactly what we say we're fine");
            sets.insert(k, bucket);
        }

        Ok(Self {
            _inner_pool: pool,
            _parent: device.clone(),
            _sets: sets,
            layout,
        })
    }

    pub fn _get_descriptor_bucket_mut(
        &mut self,
        k: &Key,
    ) -> Option<&mut Vec<DescriptorSet>> {
        self._sets.get_mut(k)
    }
    pub fn _get_descriptor_bucket(
        &self,
        k: &Key,
    ) -> Option<&Vec<DescriptorSet>> {
        self._sets.get(k)
    }

    pub fn layout_handle(&self) -> ash::vk::DescriptorSetLayout {
        self.layout.inner
    }
}

#[derive(Debug)]
struct DescriptorSetLayout {
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

#[derive(Debug)]
struct DescriptorPool {
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
