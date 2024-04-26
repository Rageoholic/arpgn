use std::{
    collections::HashSet, ffi::CStr, fmt::Debug, mem::ManuallyDrop, os::raw::c_void, ptr::null,
    sync::Arc,
};

use ash::{
    ext::debug_utils,
    khr::{self, surface, swapchain},
    vk::{
        self, CompositeAlphaFlagsKHR, DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT,
        DebugUtilsMessengerCreateInfoEXT, Extent2D, ImageUsageFlags, PhysicalDeviceType,
        PresentModeKHR, SharingMode, SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Entry, Instance, LoadingError,
};

use structopt::StructOpt;
use strum::EnumString;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

struct ContextNonDebug {
    entry: ManuallyDrop<Entry>,
    instance: Instance,
    swapchain: SwapchainKHR,
    swapchain_device: swapchain::Device,
    logger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    dev: ash::Device,
    surface: vk::SurfaceKHR,
    surface_instance: surface::Instance,
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
        //SAFETY: We need to destroy things in a specific order. First the
        //device, then the surface, then any present debug messenger, then the
        //instance, then the window, then the entry
        unsafe {
            self.nd
                .swapchain_device
                .destroy_swapchain(self.nd.swapchain, ALLOCATION_CALLBACKS.as_ref());
            self.nd.dev.destroy_device(ALLOCATION_CALLBACKS.as_ref());
            self.nd
                .surface_instance
                .destroy_surface(self.nd.surface, ALLOCATION_CALLBACKS.as_ref());
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
    SurfaceCreation,
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

        let required_instance_extensions: Vec<_> =
            ash_window::enumerate_required_extensions(win.display_handle().unwrap().into())
                .unwrap()
                .to_vec();

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|_| Error::InstanceCreation)?;

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|_| Error::InstanceCreation)?;

        let mut missing_instance_extensions = Vec::new();

        for ext in &required_instance_extensions {
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
                None => missing_instance_extensions.push(ext_cstr),
            }
        }

        if !missing_instance_extensions.is_empty() {
            log::error!(target:"graphic_subsystem", "missing mandatory extensions {:?}", missing_instance_extensions);
            return Err(Error::MissingMandatoryExtensions(
                missing_instance_extensions,
            ));
        }

        let mut instance_extensions = required_instance_extensions;

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
            instance_extensions.push(debug_utils::NAME.as_ptr());
            layers.push(KHRONOS_VALIDATION_LAYER_NAME.as_ptr());
        } else {
            log::debug!(target: "graphics_subsystem", "not inserting debug layers");
            opts.graphics_validation_layers = GraphicsValidationLevel::None;
        };

        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            enabled_extension_count: instance_extensions.len() as u32,
            ..Default::default()
        };

        let instance = {
            //SAFETY: cannot be used after entry is dropped. All pointers in the
            //create infos and associated structs must be valid. allocation
            //callbacks must be None or valid
            //
            //Allocation callbacks:
            //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
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

        //SAFETY: surface must be destroyed before win is dropped. Allocation
        //callbacks must be None or a valid allocation callback
        //
        //Allocation Callbacks
        //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                win.display_handle().unwrap().as_raw(),
                win.window_handle().unwrap().as_raw(),
                ALLOCATION_CALLBACKS.as_ref(),
            )
        }
        .map_err(|_| Error::SurfaceCreation)?;
        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        let required_instance_extensions = [khr::swapchain::NAME.as_ptr()];

        //SAFETY: this vec is dropped before instance is destroyed
        let phys_devs = unsafe { instance.enumerate_physical_devices() }.unwrap();

        let mut missing_instance_extensions = Vec::new();

        let (_, phys_dev, qfi) = phys_devs
            .iter()
            .map(|phys_dev| {
                let (score, qfp) = evaluate_phys_dev(
                    *phys_dev,
                    &instance,
                    surface,
                    &surface_instance,
                    |extension_list| {
                        for ext in required_instance_extensions {
                            //SAFETY: Lifetime of this cstr is less than
                            //lifetime of ext
                            let ext = unsafe { CStr::from_ptr(ext) };
                            if !extension_list
                                .iter()
                                .any(|dev_ext| dev_ext.extension_name_as_c_str().unwrap().eq(ext))
                            {
                                missing_instance_extensions.push(ext)
                            }
                        }
                        if missing_instance_extensions.is_empty() {
                            Some(0)
                        } else {
                            None
                        }
                    },
                );
                (score, phys_dev, qfp)
            })
            .rev()
            .max_by(|(score, _, _), (score2, _, _)| score.cmp(score2))
            .filter(|(score, _, _)| *score > 0)
            .ok_or(Error::NoSuitableDevice)?;
        let dev_extensions = required_instance_extensions;

        let queue_priorities = [1_f32];

        let mut dev_queue_indices = HashSet::with_capacity(4);

        dev_queue_indices.insert(qfi.graphics.unwrap());
        dev_queue_indices.insert(qfi.present.unwrap());

        let dev_queue_cis = dev_queue_indices
            .iter()
            .copied()
            .map(|queue_family_index| vk::DeviceQueueCreateInfo {
                queue_count: 1,
                queue_family_index,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let dev_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let dev_ci = vk::DeviceCreateInfo {
            p_queue_create_infos: dev_queue_cis.as_ptr(),
            queue_create_info_count: dev_queue_cis.len() as u32,
            p_enabled_features: &dev_features,
            pp_enabled_extension_names: dev_extensions.as_ptr(),
            enabled_extension_count: dev_extensions.len() as u32,
            ..Default::default()
        };

        let dev = {
            //Safety: We need to make sure all pointers in dev_ci are valid or
            //null, as well as for any sub-objects of dev_ci like dev_queue_cis.
            //p_user_data in the allocation callbacks must be valid until the
            //debug utils logger is destroyed
            //
            //AllocationCallbacks
            //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
            unsafe { instance.create_device(*phys_dev, &dev_ci, ALLOCATION_CALLBACKS.as_ref()) }
                .map_err(|_| Error::DeviceCreation)?
        };
        //SAFETY: Last use of graphics_queue must be before dev is destroyed,
        //family index must be within bounds
        let graphics_queue = unsafe { dev.get_device_queue(qfi.graphics.unwrap(), 0) };
        //SAFETY: Last use of graphics_queue must be before dev is destroyed,
        //family index must be within bounds
        let present_queue = unsafe { dev.get_device_queue(qfi.present.unwrap(), 0) };

        let swapchain_device = swapchain::Device::new(&instance, &dev);

        let swapchain_formats = {
            //SAFETY: Dropped before instance destroyed. Valid Surface
            unsafe { surface_instance.get_physical_device_surface_formats(*phys_dev, surface) }
        }
        .unwrap();
        //SAFETY: Dropped before instance destroyed. Valid Surface
        let swapchain_present_modes = unsafe {
            surface_instance.get_physical_device_surface_present_modes(*phys_dev, surface)
        }
        .unwrap();

        //SAFETY: Dropped before instance destroyed. Valid Surface
        let swapchain_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(*phys_dev, surface)
        }
        .unwrap();

        let swapchain_format = swapchain_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .or_else(|| swapchain_formats.iter().copied().next())
            .unwrap();

        let swapchain_present_mode = swapchain_present_modes
            .iter()
            .copied()
            .find(|present_mode| *present_mode == PresentModeKHR::MAILBOX)
            .unwrap_or(PresentModeKHR::FIFO);

        let win_size = win.inner_size();
        let swap_extent = Extent2D {
            width: win_size.width.clamp(
                swapchain_capabilities.min_image_extent.width,
                swapchain_capabilities.max_image_extent.width,
            ),
            height: win_size.height.clamp(
                swapchain_capabilities.min_image_extent.height,
                swapchain_capabilities.max_image_extent.height,
            ),
        };

        let swap_image_count = (swapchain_capabilities.min_image_count + 1)
            .min(swapchain_capabilities.max_image_count);

        let qfi_if_needed = [qfi.graphics.unwrap(), qfi.present.unwrap()];

        let swapchain_ci = SwapchainCreateInfoKHR {
            surface,
            min_image_count: swap_image_count,
            image_format: swapchain_format.format,
            image_color_space: swapchain_format.color_space,
            image_extent: swap_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: if qfi.graphics == qfi.present {
                SharingMode::EXCLUSIVE
            } else {
                SharingMode::CONCURRENT
            },
            queue_family_index_count: if qfi.graphics == qfi.present { 0 } else { 2 },
            p_queue_family_indices: if qfi.graphics == qfi.present {
                null()
            } else {
                qfi_if_needed.as_ptr()
            },
            pre_transform: swapchain_capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: swapchain_present_mode,
            clipped: vk::TRUE,

            ..Default::default()
        };

        //SAFETY: Allocation callbacks must be valid, swapchain_ci must be
        //valid, swapchain must be destroyed before surface
        let swapchain = unsafe {
            swapchain_device.create_swapchain(&swapchain_ci, ALLOCATION_CALLBACKS.as_ref())
        }
        .unwrap();

        let graphics_context = Context {
            win: ManuallyDrop::new(win),
            nd: ContextNonDebug {
                _present_queue: present_queue,
                swapchain,
                swapchain_device,
                logger,
                entry: ManuallyDrop::new(entry),
                instance,
                dev,
                _graphics_queue: graphics_queue,
                surface,
                surface_instance,
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
    present: Option<u32>,
}

impl QueueFamilyIndices {
    fn find(
        phys_dev: vk::PhysicalDevice,
        instance: &Instance,
        surface: vk::SurfaceKHR,
        surface_instance: &surface::Instance,
    ) -> Self {
        //SAFETY: Dropped before instance is destroyed
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };
        let candidate_graphics_queues = queue_families
            .iter()
            .enumerate()
            .map(|(i, qfp)| (i as u32, qfp))
            .filter(|(_, qfp)| (qfp.queue_flags.intersects(vk::QueueFlags::GRAPHICS)))
            .collect::<Vec<_>>();
        let graphics = candidate_graphics_queues
            .iter()
            //SAFETY: Due to the properties of enumerate, i is always in bounds
            .filter(|(i, _)| unsafe {
                surface_instance
                    .get_physical_device_surface_support(phys_dev, *i, surface)
                    .unwrap()
            })
            .map(|(i, _)| i)
            .next()
            .or_else(|| candidate_graphics_queues.iter().map(|(i, _)| i).next())
            .copied();

        let present = graphics
            .iter()
            //SAFETY: graphics queue index is always in bounds
            .filter(|graphics_queue_index| unsafe {
                surface_instance
                    .get_physical_device_surface_support(phys_dev, **graphics_queue_index, surface)
                    .unwrap()
            })
            .copied()
            .next()
            .or_else(|| {
                candidate_graphics_queues
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i as u32)
                    .find(|queue_family_index| {
                        //SAFETY: queue_family_index will always be in bounds due to enumerate
                        unsafe {
                            surface_instance.get_physical_device_surface_support(
                                phys_dev,
                                *queue_family_index,
                                surface,
                            )
                        }
                        .unwrap()
                    })
            });

        Self { graphics, present }
    }
}

fn evaluate_phys_dev<F: FnOnce(&[vk::ExtensionProperties]) -> Option<u32>>(
    phys_dev: vk::PhysicalDevice,
    instance: &Instance,
    surface: vk::SurfaceKHR,
    surface_instance: &surface::Instance,
    score_extensions: F,
) -> (u32, QueueFamilyIndices) {
    match score_extensions(
        //SAFETY: Last use before instance is destroyed
        &unsafe { instance.enumerate_device_extension_properties(phys_dev) }.unwrap(),
    ) {
        Some(mut score) => {
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

            let queue_family_indexes =
                QueueFamilyIndices::find(phys_dev, instance, surface, surface_instance);

            suitable &= features.geometry_shader != 0
                && queue_family_indexes.graphics.is_some()
                && queue_family_indexes.present.is_some();

            if suitable {
                //TODO: Better grading
                (score, queue_family_indexes)
            } else {
                (0, queue_family_indexes)
            }
        }
        None => (
            0,
            QueueFamilyIndices {
                graphics: None,
                present: None,
            },
        ),
    }
}
