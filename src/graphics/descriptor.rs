use std::{collections::HashMap, sync::Arc};

use ash::{prelude::VkResult, vk};

use super::Device;

pub struct DescriptorSets {
    inner_pool: vk::DescriptorPool,
    parent: Arc<Device>,
    sets: HashMap<vk::DescriptorType, Vec<vk::DescriptorSet>>,
}

impl Drop for DescriptorSets {
    fn drop(&mut self) {
        //SAFETY: We are in drop so this pool will no longer be used after this.
        //All child descriptor sets are also gone
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_descriptor_pool(self.inner_pool, None)
        };
    }
}

impl DescriptorSets {
    ///SAFETY REQUIREMENTS: valid ci
    pub unsafe fn new(
        device: &Arc<Device>,
        ci: &vk::DescriptorPoolCreateInfo,
    ) -> VkResult<Self> {
        let pool =
            //SAFETY: valid CI
            unsafe { device.as_inner_ref().create_descriptor_pool(ci, None) }?;
        let mut sets = HashMap::with_capacity(ci.pool_size_count as usize);
        //SAFETY: Because the ci is valid, we know that we can make this slice
        let pool_sizes = unsafe {
            std::slice::from_raw_parts(
                ci.p_pool_sizes,
                ci.pool_size_count as usize,
            )
        };
        for pool_size in pool_sizes {
            let set_layouts = vec![vk::DescriptorSetLaout]
            let alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&ve)
            let new_sets = device
                .as_inner_ref()
                .allocate_descriptor_sets(&alloc_info)?;
            let bucket =sets.entry(pool_size.ty)

                .or_insert_with(|| Vec::new());
            bucket.reserve(
                bucket.len() + pool_size.descriptor_count as usize,
            );
            bucket.extend()
        }

        Ok(Self {
            inner_pool: pool,
            parent: device.clone(),
        })
    }
}
