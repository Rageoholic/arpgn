use std::{ffi::CStr, fmt::Debug, mem::ManuallyDrop, os::raw::c_void, sync::Arc};

use ash::{
    ext::debug_utils,
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
    },
    Entry, Instance, LoadingError,
};
use log::Level;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

struct ContextNonDebug {
    entry: ManuallyDrop<Entry>,
    instance: ManuallyDrop<Instance>,
    logger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
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
        //SAFETY: We need to destroy things in a specific order. First any
        //present debug messenger, then the instance, then the window, then the
        //entry
        unsafe {
            if let Some((instance, logger)) = self.nd.logger.as_ref() {
                instance.destroy_debug_utils_messenger(*logger, None)
            }
            self.nd.instance.destroy_instance(None);
            ManuallyDrop::drop(&mut self.nd.instance);
            ManuallyDrop::drop(&mut self.win);
            ManuallyDrop::drop(&mut self.nd.entry);
        }
    }
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: bool,
}

#[derive(Debug)]
pub enum Error {
    Loading(LoadingError),
    InstanceCreation,
    MissingMandatoryExtensions(Vec<&'static CStr>),
}

//SAFETY: Meant to be passed to pfn_user_callback in DebugUtilsCreateInfoEXT.
//Otherwise requires that all pointers passed are valid and the object pointed
//to by callback_data has all of *it's* pointers valid
unsafe extern "system" fn debug_callback(
    sev: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> u32 {
    //SAFETY: Assuming callback_data is valid
    let callback_data = unsafe { *callback_data };

    //SAFETY: Assuming callback_data's pointers are valid
    let log_cstr = unsafe { CStr::from_ptr(callback_data.p_message) };
    //SAFETY: Assuming callback_data's pointers are valid
    let log_id_cstr = unsafe { CStr::from_ptr(callback_data.p_message_id_name) };

    let message_type = debug_utils_message_type_to_str(message_type);
    match sev {
        DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::trace!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        _ => {
            //just assume it's an error

            log::error!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
    }
    0
}

fn debug_utils_message_type_to_str(message_type: DebugUtilsMessageTypeFlagsEXT) -> &'static str {
    match message_type {
        DebugUtilsMessageTypeFlagsEXT::GENERAL => "graphics_subsystem.debug_utils.general",
        DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "graphics_subsystem.debug_utils.perf",
        DebugUtilsMessageTypeFlagsEXT::VALIDATION => "graphics_subsystem.debug_utils.validation",
        DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => {
            "graphics_subsystem.debug_utils.device_address_binding"
        }
        _ => "graphics_subsystem.debug_utils.unknown",
    }
}

const KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

impl Context {
    pub fn new(win: Arc<Window>, mut opts: ContextCreateOpts) -> Result<Self, Error> {
        //SAFETY: You may not call vulkan functions after Entry is dropped.
        //Therefore Entry should be the last thing dropped.
        let entry = unsafe { Entry::load().map_err(Error::Loading) }?;

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            p_application_name: c"placeholder".as_ptr(),
            p_engine_name: c"placeholder".as_ptr(),
            ..Default::default()
        };

        let required_extensions: Vec<_> =
            ash_window::enumerate_required_extensions(win.display_handle().unwrap().into())
                .unwrap()
                .to_vec();

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|_| Error::InstanceCreation)?;

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|_| Error::InstanceCreation)?;

        let mut missing_extensions = Vec::new();

        for ext in &required_extensions {
            //SAFETY: We know that these extensions are valid for the runtime of
            //the program because they come from ash_window which has a lifetime
            //of static
            let ext_cstr = unsafe { CStr::from_ptr(*ext) };

            match avail_extensions.iter().find(|instance_ext| {
                let instance_ext_cstr = instance_ext
                    .extension_name_as_c_str()
                    .expect("Should always be a valid cstr");

                log::trace!(
                    target: "graphics_subsystem",
                    "comparing mandatory ext {:?} to ext {:?}",
                    ext_cstr,
                    instance_ext_cstr
                );

                ext_cstr.eq(instance_ext_cstr)
            }) {
                Some(_) => {
                    log::debug!(target: "graphics_subsystem", "mandatory extension {:?} present", ext_cstr)
                }
                None => missing_extensions.push(ext_cstr),
            }
        }

        if !missing_extensions.is_empty() {
            log::error!(target:"graphic_subsystem", "missing mandatory extensions {:?}", missing_extensions);
            return Err(Error::MissingMandatoryExtensions(missing_extensions));
        }

        let mut extensions = required_extensions;

        let mut layers = Vec::new();

        if opts.graphics_validation_layers
            && avail_extensions.iter().any(|instance_ext| {
                instance_ext
                    .extension_name_as_c_str()
                    .unwrap()
                    .eq(debug_utils::NAME)
            })
            && avail_layers.iter().any(|instance_layer| {
                instance_layer
                    .layer_name_as_c_str()
                    .expect("Layer properties should all be valid c strings")
                    .eq(KHRONOS_VALIDATION_LAYER_NAME)
            })
        {
            log::debug!(
                target: "graphics_subsystem",
                "Debug layer {:?} is present and we are using it",
                KHRONOS_VALIDATION_LAYER_NAME);
            extensions.push(debug_utils::NAME.as_ptr());
            layers.push(KHRONOS_VALIDATION_LAYER_NAME.as_ptr());
        } else {
            log::debug!(target: "graphics_subsystem", "not inserting debug layers");
            opts.graphics_validation_layers = false;
        };

        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: extensions.as_ptr(),
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            enabled_extension_count: extensions.len() as u32,
            ..Default::default()
        };

        //SAFETY: cannot be used after entry is dropped. All pointers in the
        //create infos and associated structs must be valid
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|_| Error::InstanceCreation)?;

        let logger = if opts.graphics_validation_layers {
            let debug_instance = debug_utils::Instance::new(&entry, &instance);
            let create_info = DebugUtilsMessengerCreateInfoEXT {
                message_severity: log_severity_to_debug_utils_severity(),
                message_type: DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | DebugUtilsMessageTypeFlagsEXT::GENERAL,
                pfn_user_callback: Some(debug_callback),
                ..Default::default()
            };
            //SAFETY: We know this is safe because all the pointers in create_info are null or valid.
            match unsafe { debug_instance.create_debug_utils_messenger(&create_info, None) } {
                Ok(messenger) => {
                    //SAFETY: We know this is safe because all the pointers in
                    //the passed DebugUtilsMessengerCallbackDataEXT are valid or
                    //null
                    unsafe {
                        debug_instance.submit_debug_utils_message(
                            DebugUtilsMessageSeverityFlagsEXT::INFO,
                            DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                            &DebugUtilsMessengerCallbackDataEXT {
                                p_message: c"Validating Debug Messenger is working".as_ptr(),
                                p_message_id_name: c"Test".as_ptr(),
                                ..Default::default()
                            },
                        )
                    }
                    Some((debug_instance, messenger))
                }
                Err(res) => {
                    log::error!(
                        target: "graphics_subsystem",
                        "Could not create debug layer despite having ext and layer selected. VK_RESULT {:?}",
                        res
                    );
                    None
                }
            }
        } else {
            None
        };

        let graphics_context = Context {
            win: ManuallyDrop::new(win),
            nd: ContextNonDebug {
                logger,
                entry: ManuallyDrop::new(entry),
                instance: ManuallyDrop::new(instance),
            },
        };
        Ok(graphics_context)
    }
}

fn log_severity_to_debug_utils_severity() -> DebugUtilsMessageSeverityFlagsEXT {
    match log::max_level() {
        log::LevelFilter::Off => DebugUtilsMessageSeverityFlagsEXT::empty(),
        log::LevelFilter::Error => DebugUtilsMessageSeverityFlagsEXT::ERROR,
        log::LevelFilter::Warn => {
            DebugUtilsMessageSeverityFlagsEXT::WARNING | DebugUtilsMessageSeverityFlagsEXT::ERROR
        }
        log::LevelFilter::Info => {
            DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO
        }
        log::LevelFilter::Debug => {
            DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO
        }
        log::LevelFilter::Trace => {
            DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO
                | DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        }
    }
}

fn _debug_utils_severity_to_log_severity(_sev: DebugUtilsMessageSeverityFlagsEXT) {
    match _sev {
        DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
        DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
        DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Trace,
        _ => {
            //just assume it's an error
            Level::Error
        }
    };
}
