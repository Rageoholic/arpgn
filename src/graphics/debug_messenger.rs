// FILE SAFETY REQUIREMENTS
// * Only create DebugMessenger in DebugMessenger::new

// * Do not destroy inner except in the Drop implementation

// * We must hold an Arc to our parent instance

use std::ffi::{c_void, CStr};

use ash::{
    ext::debug_utils,
    prelude::VkResult,
    vk::{
        DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
        DebugUtilsMessengerEXT,
    },
};

use super::Instance;

pub(super) struct DebugMessenger {
    inner: DebugUtilsMessengerEXT,
    instance: debug_utils::Instance,
}
impl std::fmt::Debug for DebugMessenger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugMessenger")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}
impl Drop for DebugMessenger {
    fn drop(&mut self) {
        //SAFETY: We know that our inner was made from the debug_utils::Instance
        //coupled with us
        unsafe {
            self.instance
                .destroy_debug_utils_messenger(self.inner, None)
        };
    }
}

impl DebugMessenger {
    //SAFETY REQUIREMENTS: Valid ci
    pub(super) unsafe fn new(
        parent_instance: &Instance,
        ci: &DebugUtilsMessengerCreateInfoEXT,
    ) -> VkResult<Self> {
        let debug_utils_instance =
            debug_utils::Instance::new(parent_instance.parent(), parent_instance.as_inner_ref());
        //SAFETY: valid ci. We know because precondition of this unsafe function
        unsafe { debug_utils_instance.create_debug_utils_messenger(ci, None) }.map(|inner| Self {
            inner,
            instance: debug_utils_instance,
        })
    }

    //This is useful and I don't want the compiler whining at me
    #[allow(dead_code)]
    pub(super) fn send_message(
        &self,
        message_severity: DebugUtilsMessageSeverityFlagsEXT,
        message_types: DebugUtilsMessageTypeFlagsEXT,
        message: &CStr,
    ) {
        let callback_data = DebugUtilsMessengerCallbackDataEXT::default().message(message);
        //SAFETY: We made callback_data ourselves
        unsafe {
            self.instance.submit_debug_utils_message(
                message_severity,
                message_types,
                &callback_data,
            )
        };
    }
}

//SAFETY: Meant to be passed to pfn_user_callback in DebugUtilsCreateInfoEXT.
//Otherwise requires that all pointers passed are valid and the object pointed
//to by callback_data has all of *it's* pointers valid
pub(super) unsafe extern "system" fn default_debug_callback(
    sev: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> u32 {
    //SAFETY: Assuming callback_data is valid
    let callback_data = unsafe { *callback_data };

    //SAFETY: Assuming callback_data's pointers are valid
    let log_cstr = unsafe { CStr::from_ptr(callback_data.p_message) };
    let log_id_cstr = if !callback_data.p_message_id_name.is_null() {
        //SAFETY: Assuming callback_data's pointers are valid
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }
    } else {
        c"{null}"
    };

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
