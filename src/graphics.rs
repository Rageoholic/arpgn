use std::{ffi::CStr, fmt::Debug, mem::ManuallyDrop, sync::Arc};

use ash::{vk, Entry, Instance, LoadingError};
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

struct ContextNonDebug {
    entry: ManuallyDrop<Entry>,
    _instance: ManuallyDrop<Instance>,
}

impl Debug for ContextNonDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsContextNonDebug")
            .finish_non_exhaustive()
    }
}

//SAFETY: All members must be manually drop so we can control the Drop order in
//our Drop implementation. There are ways around this but they require more
//magic
#[derive(Debug)]
pub struct Context {
    win: ManuallyDrop<Arc<Window>>,
    nd: ContextNonDebug,
}

impl Drop for Context {
    fn drop(&mut self) {
        //SAFETY: Cannot use self.win past this point (shouldn't be a problem
        //but hey)
        unsafe { ManuallyDrop::drop(&mut self.win) }

        unsafe { ManuallyDrop::drop(&mut self.nd.entry) }
    }
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: bool,
}

#[derive(Debug)]
pub enum Error {
    LoadingError(LoadingError),
    InstanceError,
}

impl Context {
    pub fn new(win: Arc<Window>, _opts: ContextCreateOpts) -> Result<Self, Error> {
        //SAFETY: You may not call vulkan functions after Entry is dropped.
        //Therefore Entry should be the last thing dropped.
        let entry = unsafe { Entry::load().map_err(|err| Error::LoadingError(err)) }?;

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            p_application_name: c"placeholder".as_ptr(),
            p_engine_name: c"placeholder".as_ptr(),
            ..Default::default()
        };

        let required_extensions: Vec<_> =
            ash_window::enumerate_required_extensions(win.display_handle().unwrap().into())
                .unwrap()
                .iter()
                .copied()
                .collect();

        log::info!(target: "graphics-subsystem", "required extensions: {:?}", required_extensions.iter().copied().map(|p|unsafe {CStr::from_ptr(p)}).collect::<Vec<_>>());

        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            enabled_extension_count: required_extensions.len() as u32,
            ..Default::default()
        };

        //SAFETY: cannot be used after entry is dropped. All pointers in the
        //create infos and associated structs must be valid
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|_| Error::InstanceError)?;

        let graphics_context = Context {
            win: ManuallyDrop::new(win),
            nd: ContextNonDebug {
                entry: ManuallyDrop::new(entry),
                _instance: ManuallyDrop::new(instance),
            },
        };
        Ok(graphics_context)
    }
}
