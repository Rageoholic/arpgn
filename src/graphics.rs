use std::{ffi::CStr, fmt::Debug, mem::ManuallyDrop, os::raw::c_void, sync::Arc};

use ash::{
    ext::debug_utils,
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT, PhysicalDeviceType,
    },
    Entry, Instance, LoadingError,
};
use log::Level;
use structopt::StructOpt;
use strum::EnumString;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

struct ContextNonDebug {
    entry: ManuallyDrop<Entry>,
    instance: Instance,
    logger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    _graphics_queue: vk::Queue,
    dev: ash::Device,
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
static ALLOCATION_CALLBACKS: Option<vk::AllocationCallbacks> = None;

impl Drop for Context {
    fn drop(&mut self) {
        //SAFETY: We need to destroy things in a specific order. First any
        //present debug messenger, then the instance, then the window, then the
        //entry
        unsafe {
            self.nd.dev.destroy_device(ALLOCATION_CALLBACKS.as_ref());
            if let Some((instance, logger)) = self.nd.logger.as_ref() {
                instance.destroy_debug_utils_messenger(*logger, ALLOCATION_CALLBACKS.as_ref())
            }
            self.nd
                .instance
                .destroy_instance(ALLOCATION_CALLBACKS.as_ref());
            ManuallyDrop::drop(&mut self.win);
            ManuallyDrop::drop(&mut self.nd.entry);
        }
    }
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: GraphicsValidationLevel,
}

#[derive(Debug, StructOpt, Default, PartialEq, Eq, EnumString, Clone, Copy)]
pub enum GraphicsValidationLevel {
    #[default]
    None,
    Error,
    Warning,
    Info,
    Verbose,
}

#[derive(Debug)]
pub enum Error {
    Loading(LoadingError),
    InstanceCreation,
    MissingMandatoryExtensions(Vec<&'static CStr>),
    NoSuitableDevice,
    DeviceCreation,
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
    //TODO: Split this function the fuck up. Christ this is long
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

        if opts.graphics_validation_layers != GraphicsValidationLevel::None
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
            opts.graphics_validation_layers = GraphicsValidationLevel::None;
        };

        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: extensions.as_ptr(),
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            enabled_extension_count: extensions.len() as u32,
            ..Default::default()
        };

        let instance = {
            //SAFETY: cannot be used after entry is dropped. All pointers in the
            //create infos and associated structs must be valid. allocation
            //callbacks must be None or p_user_data must be valid for the
            //lifetime of the instance
            unsafe { entry.create_instance(&instance_ci, ALLOCATION_CALLBACKS.as_ref()) }
                .map_err(|_| Error::InstanceCreation)?
        };

        let logger = if opts.graphics_validation_layers != GraphicsValidationLevel::None {
            let debug_instance = debug_utils::Instance::new(&entry, &instance);
            let create_info = DebugUtilsMessengerCreateInfoEXT {
                message_severity: graphics_validation_sev_to_debug_utils_flags(
                    opts.graphics_validation_layers,
                ),
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

        //SAFETY: this vec is dropped before instance is destroyed
        let phys_devs = unsafe { instance.enumerate_physical_devices() }.unwrap();

        let (_, phys_dev, qfi) = phys_devs
            .iter()
            .map(|phys_dev| {
                let (score, qfp) = evaluate_phys_dev(*phys_dev, &instance);
                (score, phys_dev, qfp)
            })
            .rev()
            .max_by(|(score, _, _), (score2, _, _)| score.cmp(score2))
            .filter(|(score, _, _)| *score > 0)
            .ok_or(Error::NoSuitableDevice)?;

        let queue_priorities = [1_f32];

        let dev_queue_cis = [vk::DeviceQueueCreateInfo {
            queue_family_index: qfi.graphics.unwrap(),
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let dev_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let dev_ci = vk::DeviceCreateInfo {
            p_queue_create_infos: dev_queue_cis.as_ptr(),
            queue_create_info_count: dev_queue_cis.len() as u32,
            p_enabled_features: &dev_features,
            ..Default::default()
        };

        let dev = {
            //Safety: We need to make sure all pointers in dev_ci are valid or
            //null, as well as for any sub-objects of dev_ci like dev_queue_cis.
            //p_user_data in the allocation callbacks must be valid until the
            //debug utils logger is destroyed
            unsafe { instance.create_device(*phys_dev, &dev_ci, ALLOCATION_CALLBACKS.as_ref()) }
                .map_err(|_| Error::DeviceCreation)?
        };
        //SAFETY: Last use of graphics_queue must be before dev is destroyed
        let graphics_queue = unsafe { dev.get_device_queue(qfi.graphics.unwrap(), 0) };

        let graphics_context = Context {
            win: ManuallyDrop::new(win),
            nd: ContextNonDebug {
                logger,
                entry: ManuallyDrop::new(entry),
                instance,
                dev,
                _graphics_queue: graphics_queue,
            },
        };
        Ok(graphics_context)
    }
}

fn graphics_validation_sev_to_debug_utils_flags(
    graphics_validation_layers: GraphicsValidationLevel,
) -> DebugUtilsMessageSeverityFlagsEXT {
    let none = DebugUtilsMessageSeverityFlagsEXT::empty();
    let error = none | DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let warning = error | DebugUtilsMessageSeverityFlagsEXT::WARNING;
    let info = warning | DebugUtilsMessageSeverityFlagsEXT::INFO;
    let verbose = info | DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
    match graphics_validation_layers {
        GraphicsValidationLevel::None => none,
        GraphicsValidationLevel::Error => error,
        GraphicsValidationLevel::Warning => warning,
        GraphicsValidationLevel::Info => info,
        GraphicsValidationLevel::Verbose => verbose,
    }
}

struct QueueFamilyIndices {
    graphics: Option<u32>,
}

impl QueueFamilyIndices {
    fn find(phys_dev: vk::PhysicalDevice, instance: &Instance) -> Self {
        //SAFETY: Dropped before instance is destroyed
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };
        let graphics = queue_families
            .iter()
            .enumerate()
            .find(|(_, qfp)| (qfp.queue_flags.intersects(vk::QueueFlags::GRAPHICS)))
            .map(|(i, _)| i as u32);
        Self { graphics }
    }
}

fn evaluate_phys_dev(
    phys_dev: vk::PhysicalDevice,
    instance: &Instance,
) -> (u32, QueueFamilyIndices) {
    let mut score = 0;
    //SAFETY: We discard features before instance is destroyed
    let features = unsafe { instance.get_physical_device_features(phys_dev) };
    //SAFETY: We discard features before instance is destroyed
    let props = unsafe { instance.get_physical_device_properties(phys_dev) };

    score += match props.device_type {
        PhysicalDeviceType::DISCRETE_GPU => 100,
        PhysicalDeviceType::INTEGRATED_GPU => 50,
        PhysicalDeviceType::VIRTUAL_GPU => 25,
        _ => 10,
    };

    let mut suitable = true;

    let queue_family_indexes = QueueFamilyIndices::find(phys_dev, instance);

    suitable &= features.geometry_shader != 0 && queue_family_indexes.graphics.is_some();

    if suitable {
        //TODO: Better grading
        (score, queue_family_indexes)
    } else {
        (0, queue_family_indexes)
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
