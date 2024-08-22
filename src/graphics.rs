use std::{ffi::CStr, fmt::Debug, rc::Rc, str::FromStr};

use ash::{vk, LoadingError};
use debug_messenger::DebugMessenger;
use instance::Instance;
use structopt::StructOpt;
use strum::EnumString;

use winit::{raw_window_handle::HasDisplayHandle, window::Window};

mod debug_messenger;
mod instance;

const _MAX_FRAMES_IN_FLIGHT: usize = 2;

struct ContextNonDebug {
    _entry: Rc<ash::Entry>,
    _instance: Rc<Instance>,
    _debug_messenger: Option<DebugMessenger>,
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
    _win: Rc<Window>,
    _nd: ContextNonDebug,
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: ValidationLevel,
}

#[derive(Debug, StructOpt, Default, PartialEq, Eq, EnumString, Clone, Copy)]
pub enum ValidationLevel {
    #[default]
    None,
    Error,
    Warning,
    Info,
    Verbose,
}

#[derive(Debug)]
pub enum Error {
    #[allow(dead_code)]
    Loading(LoadingError),
    InstanceCreation,
    #[allow(dead_code)]
    _MissingMandatoryExtensions(Vec<String>),
    _NoSuitableDevice,
    _DeviceCreation,
    _SurfaceCreation,
    _SwapchainCreation,
    _CommandBufferCreation,
    #[allow(dead_code)]
    Unknown(Result<String, std::convert::Infallible>),
}

const KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(
        win: &Rc<Window>,
        mut opts: ContextCreateOpts,
    ) -> Result<Self, Error> {
        let entry =
        //SAFETY: Must be dropped *after* instance, which we accomplish by
        //having instance hold on to a ref counted pointer to entry
            Rc::new(unsafe { ash::Entry::load().map_err(Error::Loading) }?);
        //SAFETY: Should be good?
        let vk_version = match unsafe { entry.try_enumerate_instance_version() }
        {
            Ok(ver) => ver.unwrap_or(vk::API_VERSION_1_0),

            Err(_) => unreachable!(),
        };
        log::info!(
            "Available vk version {}.{}.{}",
            vk::api_version_major(vk_version),
            vk::api_version_minor(vk_version),
            vk::api_version_patch(vk_version)
        );
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk_version)
            .application_name(c"arpgn");
        let avail_extensions =
        //SAFETY: Should always be good
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|_| {
                    Error::Unknown(String::from_str(
                        "Couldn't load instance extensions",
                    ))
                }).unwrap();
        let windowing_required_extensions =
            ash_window::enumerate_required_extensions(
                win.display_handle().unwrap().as_raw(),
            )
            .unwrap();
        let mut mandatory_extensions: Vec<&CStr> =
            Vec::with_capacity(windowing_required_extensions.len());
        mandatory_extensions.extend(
            windowing_required_extensions
                .iter()
                //SAFETY: ash_window will give us back c strings. If an
                //extension name is longer than isize::MAX I will eat my
                //computer
                .map(|bytes| unsafe { CStr::from_ptr(*bytes) }),
        );

        let mut missing_mandatory_extensions =
            Vec::with_capacity(mandatory_extensions.len());

        for ext in &mandatory_extensions {
            let mut found = false;
            for instance_ext in &avail_extensions {
                if (*ext).eq(instance_ext.extension_name_as_c_str().unwrap()) {
                    found = true;
                    break;
                }
            }
            if !found {
                missing_mandatory_extensions
                    .push(ext.to_str().unwrap().to_owned());
            }
        }

        if !missing_mandatory_extensions.is_empty() {
            return Err(Error::_MissingMandatoryExtensions(
                missing_mandatory_extensions,
            ));
        }
        let mut instance_exts: Vec<_> = mandatory_extensions
            .iter()
            .map(|str| str.as_ptr())
            .collect();
        if opts.graphics_validation_layers != ValidationLevel::None
            && avail_extensions
                .iter()
                .find_map(|instance_ext| {
                    if ash::ext::debug_utils::NAME
                        .eq(instance_ext.extension_name_as_c_str().unwrap())
                    {
                        Some(())
                    } else {
                        None
                    }
                })
                .is_some()
        {
            instance_exts.push(ash::ext::debug_utils::NAME.as_ptr());
        } else {
            opts.graphics_validation_layers = ValidationLevel::None;
        }

        let mut layer_names = Vec::with_capacity(1);
        if opts.graphics_validation_layers != ValidationLevel::None {
            layer_names.push(KHRONOS_VALIDATION_LAYER_NAME.as_ptr());
        }

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_exts)
            .enabled_layer_names(&layer_names);

        let all_debug_message_types =
            vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

        let mut debug_utils_messenger_ci =
            vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(graphics_validation_sev_to_debug_utils_flags(
                    opts.graphics_validation_layers,
                ))
                .message_type(all_debug_message_types)
                .pfn_user_callback(Some(
                    debug_messenger::default_debug_callback,
                ));
        if !layer_names.is_empty() {
            instance_create_info =
                instance_create_info.push_next(&mut debug_utils_messenger_ci);
        }
        let instance = Rc::new(
            //SAFETY: Valid ci. We know because we made it and none of the lifetimes
            //involved have expired
            unsafe { Instance::new(&entry, &instance_create_info) }
                .map_err(|_| Error::InstanceCreation)?,
        );

        let debug_messenger = if opts.graphics_validation_layers
            != ValidationLevel::None
        {
            //SAFETY: Valid ci. We know cause we made it
            unsafe { DebugMessenger::new(&instance, &debug_utils_messenger_ci) }
                .ok()
        } else {
            None
        };

        Ok(Context {
            _win: win.clone(),
            _nd: ContextNonDebug {
                _entry: entry,
                _instance: instance,
                _debug_messenger: debug_messenger,
            },
        })
    }
    pub fn resize(&mut self) {}
    pub fn draw(&mut self) {}
}

fn graphics_validation_sev_to_debug_utils_flags(
    graphics_validation_layers: ValidationLevel,
) -> vk::DebugUtilsMessageSeverityFlagsEXT {
    let none = vk::DebugUtilsMessageSeverityFlagsEXT::empty();
    let error = none | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let warning = error | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    let info = warning | vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
    let verbose = info | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
    match graphics_validation_layers {
        ValidationLevel::None => none,
        ValidationLevel::Error => error,
        ValidationLevel::Warning => warning,
        ValidationLevel::Info => info,
        ValidationLevel::Verbose => verbose,
    }
}
