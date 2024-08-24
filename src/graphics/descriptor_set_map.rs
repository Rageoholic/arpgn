// FILE SAFETY REQUIREMENTS:
// * Users must not destroy the pool or the layout handle.
// * Only create a DescriptionMap in its new function
// * Only create a DescriptionSetLayout in its new function
// * We can only destroy vk handles in drop
// * We hold an Arc to the device that created us

use std::{collections::HashMap, hash::Hash, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{
        self, DescriptorPoolSize, DescriptorSetAllocateInfo,
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
};

use super::Device;

#[derive(Debug)]
pub struct DescriptorSetMap<Key>
where
    Key: Hash + Eq,
{
    inner_pool: vk::DescriptorPool,
    parent: Arc<Device>,
    _sets: HashMap<Key, Vec<vk::DescriptorSet>>,
    layout: vk::DescriptorSetLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorRequest {
    pub count: u32,
    pub binding: u32,
    pub ty: DescriptorType,
}

impl<T> Drop for DescriptorSetMap<T>
where
    T: Hash + Eq,
{
    fn drop(&mut self) {
        //SAFETY: We are in drop so this pool will no longer be used after this.
        //All child descriptor sets are also gone
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_descriptor_set_layout(self.layout, None);
            self.parent
                .as_inner_ref()
                .destroy_descriptor_pool(self.inner_pool, None)
        };
    }
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
        //SAFETY: Valid ci
        let layout = unsafe {
            device
                .as_inner_ref()
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
        }?;
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

        let ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&descriptor_pool_sizes);

        let pool =
            //SAFETY: valid CI
            unsafe { device.as_inner_ref().create_descriptor_pool(&ci, None) }?;
        let mut sets = HashMap::with_capacity(descriptor_requests.len());

        for (k, request) in descriptor_requests {
            let duped_layouts = vec![layout; request.count as usize];
            let ai = DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&duped_layouts)
                .to_owned();
            let bucket =
                //SAFETY: We know this is good because we created the descriptor_layout
                unsafe { device.as_inner_ref().allocate_descriptor_sets(&ai) }.expect("They basically guarantee that if we never free and allocate exactly what we say we're fine");
            sets.insert(k, bucket);
        }

        Ok(Self {
            inner_pool: pool,
            parent: device.clone(),
            _sets: sets,
            layout,
        })
    }

    pub fn _get_descriptor_bucket_mut(
        &mut self,
        k: &Key,
    ) -> Option<&mut Vec<vk::DescriptorSet>> {
        self._sets.get_mut(k)
    }
    pub fn _get_descriptor_bucket(
        &self,
        k: &Key,
    ) -> Option<&Vec<vk::DescriptorSet>> {
        self._sets.get(k)
    }

    pub fn layout_handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}
