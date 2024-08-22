use std::{
    ffi::{c_void, CStr},
    rc::Rc,
};

use ash::{
    ext::debug_utils,
    prelude::VkResult,
    vk::{self, DebugUtilsMessengerCreateInfoEXT},
};

use super::Instance;

pub(super) struct DebugMessenger {
    inner: vk::DebugUtilsMessengerEXT,
    _parent: Rc<Instance>,
    instance: debug_utils::Instance,
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
        parent_instance: &Rc<Instance>,
        ci: &DebugUtilsMessengerCreateInfoEXT,
    ) -> VkResult<Self> {
        let debug_utils_instance = debug_utils::Instance::new(
            parent_instance.parent(),
            parent_instance.as_inner(),
        );
        //SAFETY: valid ci. We know because precondition of this unsafe function
        unsafe { debug_utils_instance.create_debug_utils_messenger(ci, None) }
            .map(|inner| Self {
                inner,
                _parent: parent_instance.clone(),
                instance: debug_utils_instance,
            })
    }

    //This is useful and I don't want the compiler whining at me
    #[allow(dead_code)]
    pub(super) fn send_message(
        &self,
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        message: &CStr,
    ) {
        let callback_data =
            vk::DebugUtilsMessengerCallbackDataEXT::default().message(message);
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
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
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

fn debug_utils_message_type_to_str(
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
