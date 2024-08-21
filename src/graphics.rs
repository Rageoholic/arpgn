use std::{
    ffi::{c_void, CStr},
    fmt::Debug,
    rc::Rc,
};

use ash::{vk, LoadingError};
use structopt::StructOpt;
use strum::EnumString;

use winit::window::Window;

const _MAX_FRAMES_IN_FLIGHT: usize = 2;

struct ContextNonDebug {}

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
    pub _graphics_validation_layers: ValidationLevel,
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
    _Loading(LoadingError),
    _InstanceCreation,
    #[allow(dead_code)]
    _MissingMandatoryExtensions(Vec<String>),
    _NoSuitableDevice,
    _DeviceCreation,
    _SurfaceCreation,
    _SwapchainCreation,
    _CommandBufferCreation,
}

//SAFETY: Meant to be passed to pfn_user_callback in DebugUtilsCreateInfoEXT.
//Otherwise requires that all pointers passed are valid and the object pointed
//to by callback_data has all of *it's* pointers valid
unsafe extern "system" fn _debug_callback(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> u32 {
    //SAFETY: Assuming callback_data is valid
    let callback_data = unsafe { *callback_data };

    //SAFETY: Assuming callback_data's pointers are valid
    let log_cstr = unsafe { CStr::from_ptr(callback_data.p_message) };
    //SAFETY: Assuming callback_data's pointers are valid
    let log_id_cstr =
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) };

    let message_type = _debug_utils_message_type_to_str(message_type);
    match sev {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!(
                target:message_type,
                "{:?}: {:?}",
                log_id_cstr, log_cstr
            )
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
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

fn _debug_utils_message_type_to_str(
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
) -> &'static str {
    match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => {
            "graphics_subsystem.debug_utils.general"
        }
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => {
            "graphics_subsystem.debug_utils.perf"
        }
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => {
            "graphics_subsystem.debug_utils.validation"
        }
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => {
            "graphics_subsystem.debug_utils.device_address_binding"
        }
        _ => "graphics_subsystem.debug_utils.unknown",
    }
}

const _KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(
        win: &Rc<Window>,
        _opts: ContextCreateOpts,
    ) -> Result<Self, Error> {
        //SAFETY: You may not call vulkan functions after Entry is dropped.
        //Therefore Entry should be the last thing dropped.

        Ok(Context {
            _win: win.clone(),
            _nd: ContextNonDebug {},
        })
    }
    pub fn resize(&mut self) {}
    pub fn draw(&mut self) {}
}

fn _graphics_validation_sev_to_debug_utils_flags(
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
