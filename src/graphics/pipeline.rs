use std::sync::Arc;

use ash::{
    prelude::VkResult,
    vk::{GraphicsPipelineCreateInfo, Pipeline as RawPipeline, PipelineCache},
};

use super::{utils::associate_debug_name, Device};

#[derive(Debug)]
pub struct Pipeline {
    inner: RawPipeline,
    parent: Arc<Device>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        //SAFETY: We know that parent is the device we were created from
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_pipeline(self.inner, None)
        };
    }
}

impl Pipeline {
    pub unsafe fn new_graphics_pipelines(
        device: &Arc<Device>,
        ci: &[GraphicsPipelineCreateInfo],
        mut f: Option<impl FnMut(usize, ash::vk::Pipeline) -> Option<String>>,
    ) -> VkResult<Vec<Self>> {
        //SAFETY: Valid cis
        let inners = unsafe {
            device.as_inner_ref().create_graphics_pipelines(
                PipelineCache::null(),
                ci,
                None,
            )
        }
        .map_err(|(pipelines, result)| {
            for pipeline in pipelines {
                //SAFETY: Never going to use these
                unsafe {
                    device.as_inner_ref().destroy_pipeline(pipeline, None)
                }
            }
            result
        })?;

        Ok(inners
            .iter()
            .copied()
            .enumerate()
            .map(|(i, inner)| {
                if let Some(ref mut f) = f {
                    let debug_name = f(i, inner);
                    associate_debug_name!(device, inner, debug_name)
                }
                Self {
                    inner,
                    parent: device.clone(),
                }
            })
            .collect())
    }

    pub(crate) fn get_inner(&self) -> RawPipeline {
        self.inner
    }
}
