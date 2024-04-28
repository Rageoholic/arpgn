use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    fmt::Debug,
    os::raw::c_void,
    ptr::null,
    sync::Arc,
};

use ash::{
    ext::debug_utils,
    khr::{self, surface, swapchain},
    prelude::VkResult,
    vk::{
        self, CompositeAlphaFlagsKHR, DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT,
        DebugUtilsMessengerCreateInfoEXT, Extent2D, ImageUsageFlags, PhysicalDeviceType,
        PresentModeKHR, QueueFamilyProperties, SharingMode, SurfaceCapabilitiesKHR,
        SwapchainCreateInfoKHR,
    },
    Entry, LoadingError,
};

use structopt::StructOpt;
use strum::EnumString;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

struct ContextNonDebug {
    _entry: Arc<Entry>,
    _instance: Arc<Instance>,
    _swapchain: Swapchain,
    _dev: Arc<Device>,
    _surface: Arc<Surface>,
    _logger: Option<DebugMessenger>,
}

struct Surface {
    inner: vk::SurfaceKHR,
    instance: surface::Instance,
    _parent_instance: Arc<Instance>,
    parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .destroy_surface(self.inner, ALLOCATION_CALLBACKS.as_ref())
        }
    }
}

struct PhysicalDevice {
    inner: vk::PhysicalDevice,
    _parent: Arc<Instance>,
}

impl Surface {
    fn get_physical_device_surface_support(&self, phys_dev: &PhysicalDevice, i: u32) -> bool {
        unsafe {
            self.instance
                .get_physical_device_surface_support(phys_dev.inner, i, self.inner)
                .unwrap()
        }
    }

    fn get_physical_device_surface_capabilities(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> SurfaceCapabilitiesKHR {
        //SAFETY: Copying owned data
        unsafe {
            self.instance
                .get_physical_device_surface_capabilities(phys_dev.inner, self.inner)
        }
        .unwrap()
    }

    fn get_physical_device_surface_present_modes(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> Vec<PresentModeKHR> {
        //SAFETY: fetching owned data
        unsafe {
            self.instance
                .get_physical_device_surface_present_modes(phys_dev.inner, self.inner)
        }
        .unwrap()
    }
    fn get_physical_device_surface_formats(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        //SAFETY: This is okay because surface is known to be valid and we're getting a Vec of PODs
        unsafe {
            self.instance
                .get_physical_device_surface_formats(phys_dev.inner, self.inner)
        }
    }

    fn new(instance: &Arc<Instance>, win: &Arc<Window>) -> Result<Surface, Error> {
        let surface = unsafe {
            ash_window::create_surface(
                &instance.parent,
                &instance.inner,
                win.display_handle().unwrap().as_raw(),
                win.window_handle().unwrap().as_raw(),
                ALLOCATION_CALLBACKS.as_ref(),
            )
        }
        .map_err(|_| Error::SurfaceCreation)?;
        let surface_instance = ash::khr::surface::Instance::new(&instance.parent, &instance.inner);

        Ok(Self {
            inner: surface,
            instance: surface_instance,
            _parent_instance: instance.clone(),
            parent_window: win.clone(),
        })
    }
}

struct Instance {
    inner: ash::Instance,
    parent: Arc<Entry>,
}

struct Queue {
    qfi: u32,
    _inner: vk::Queue,
    _parent: Arc<Device>,
}

struct Device {
    inner: ash::Device,
    parent: Arc<Instance>,
}

impl Device {
    fn get_device_queue(self: &Arc<Self>, qfi: u32, queue_index: u32) -> Queue {
        Queue {
            _inner: unsafe { self.inner.get_device_queue(qfi, queue_index) },
            _parent: self.clone(),
            qfi,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //SAFETY: Last use of device. All other references should have been dropped
        unsafe { self.inner.destroy_device(ALLOCATION_CALLBACKS.as_ref()) };
    }
}

impl Instance {
    fn get_physical_device_features(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> vk::PhysicalDeviceFeatures {
        unsafe { self.inner.get_physical_device_features(phys_dev.inner) }
    }

    fn new(
        entry: Arc<Entry>,

        required_instance_extensions: &[&str],
        opts: &mut ContextCreateOpts,
    ) -> Result<Self, Error> {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            p_application_name: c"placeholder".as_ptr(),
            p_engine_name: c"placeholder".as_ptr(),
            ..Default::default()
        };

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|_| Error::InstanceCreation)?;

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|_| Error::InstanceCreation)?;

        let mut missing_instance_extensions = Vec::new();

        for ext in required_instance_extensions.iter().copied() {
            //SAFETY: We know that these extensions are valid for the runtime of
            //the program because they come from ash_window which has a lifetime
            //of static

            match avail_extensions.iter().find(|instance_ext| {
                let instance_ext = instance_ext
                    .extension_name_as_c_str()
                    .expect("Should always be a valid cstr")
                    .to_str()
                    .unwrap()
                    .to_owned();

                log::trace!(
                    target: "graphics_subsystem",
                    "comparing mandatory ext {:?} to ext {:?}",
                    ext,
                    instance_ext
                );
                instance_ext == *ext
            }) {
                Some(_) => {
                    log::debug!(target: "graphics_subsystem", "mandatory extension {:?} present", ext)
                }
                None => missing_instance_extensions.push(ext.to_owned()),
            }
        }

        if !missing_instance_extensions.is_empty() {
            log::error!(target:"graphic_subsystem", "missing mandatory extensions {:?}", missing_instance_extensions);
            return Err(Error::MissingMandatoryExtensions(
                missing_instance_extensions,
            ));
        }

        let instance_extensions = required_instance_extensions
            .iter()
            .map(|s| CString::new(*s).unwrap())
            .collect::<Vec<_>>();

        let mut instance_extensions: Vec<_> =
            instance_extensions.iter().map(|s| s.as_ptr()).collect();

        let mut layers = Vec::new();

        if opts.graphics_validation_layers != ValidationLevel::None
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
            opts.graphics_validation_layers = ValidationLevel::None;
        };

        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            enabled_extension_count: instance_extensions.len() as u32,
            ..Default::default()
        };

        //SAFETY: cannot be used after entry is dropped. All pointers in the
        //create infos and associated structs must be valid. allocation
        //callbacks must be None or valid
        //
        //Allocation callbacks:
        //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
        unsafe { entry.create_instance(&instance_ci, ALLOCATION_CALLBACKS.as_ref()) }
            .map(|i| Instance {
                inner: i,
                parent: entry,
            })
            .map_err(|_| Error::InstanceCreation)
    }

    fn get_physical_device_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        unsafe { self.inner.get_physical_device_properties(phys_dev.inner) }
    }

    fn get_physical_device_queue_family_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> Vec<QueueFamilyProperties> {
        unsafe {
            self.inner
                .get_physical_device_queue_family_properties(phys_dev.inner)
        }
    }
    fn enumerate_device_extension_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> Vec<vk::ExtensionProperties> {
        unsafe {
            self.inner
                .enumerate_device_extension_properties(phys_dev.inner)
        }
        .unwrap()
    }
    fn create_device(
        self: &Arc<Self>,
        phys_dev: &PhysicalDevice,
        qfi: &QueueFamilyIndices,
        required_device_extensions: &[&str],
    ) -> Result<Device, Error> {
        let queue_priorities = [1_f32];

        let mut dev_queue_indices = HashSet::with_capacity(4);

        dev_queue_indices.insert(qfi.graphics);
        dev_queue_indices.insert(qfi.present);

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

        let required_device_extensions = required_device_extensions
            .iter()
            .map(|ext_name| CString::new(&ext_name[..]).unwrap())
            .collect::<Vec<_>>();

        let required_device_extensions = required_device_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let dev_ci = vk::DeviceCreateInfo {
            p_queue_create_infos: dev_queue_cis.as_ptr(),
            queue_create_info_count: dev_queue_cis.len() as u32,
            p_enabled_features: &dev_features,
            pp_enabled_extension_names: required_device_extensions.as_ptr(),
            enabled_extension_count: required_device_extensions.len() as u32,
            ..Default::default()
        };
        unsafe {
            self.inner
                .create_device(phys_dev.inner, &dev_ci, ALLOCATION_CALLBACKS.as_ref())
                .map(|d| Device {
                    inner: d,
                    parent: self.clone(),
                })
                .map_err(|_| Error::DeviceCreation)
        }
    }

    fn enumerate_physical_devices(self: &Arc<Self>) -> Vec<PhysicalDevice> {
        //SAFETY: this vec is dropped before instance is destroyed
        unsafe { self.inner.enumerate_physical_devices() }
            .unwrap()
            .iter()
            .map(|pd| PhysicalDevice {
                _parent: self.clone(),
                inner: *pd,
            })
            .collect()
    }
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
    _win: Arc<Window>,
    _nd: ContextNonDebug,
}
static ALLOCATION_CALLBACKS: Option<vk::AllocationCallbacks> = None;

impl Drop for Instance {
    fn drop(&mut self) {
        //SAFETY: This is always valid
        unsafe { self.inner.destroy_instance(ALLOCATION_CALLBACKS.as_ref()) };
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        //SAFETY: Always valid
        unsafe {
            self.instance
                .destroy_debug_utils_messenger(self.inner, ALLOCATION_CALLBACKS.as_ref())
        }
    }
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
    Loading(LoadingError),
    InstanceCreation,
    MissingMandatoryExtensions(Vec<String>),
    NoSuitableDevice,
    DeviceCreation,
    SurfaceCreation,
    SwapchainCreation,
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

struct DebugMessenger {
    inner: vk::DebugUtilsMessengerEXT,
    _parent: Arc<Instance>,
    instance: debug_utils::Instance,
}

struct Swapchain {
    inner: vk::SwapchainKHR,
    _parent: Arc<Device>,
    device: swapchain::Device,
    _graphics_queue: Queue,
    _present_queue: Queue,
}

impl Swapchain {
    fn new(
        device: Arc<Device>,
        phys_dev: &PhysicalDevice,
        surface: Arc<Surface>,
        graphics_queue: Queue,
        present_queue: Queue,
    ) -> Result<Self, Error> {
        let swapchain_device = swapchain::Device::new(&device.parent.inner, &device.inner);

        let swapchain_formats = { surface.get_physical_device_surface_formats(phys_dev) }.unwrap();

        let swapchain_present_modes = surface.get_physical_device_surface_present_modes(phys_dev);

        let swapchain_capabilities = surface.get_physical_device_surface_capabilities(phys_dev);

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

        let win_size = surface.parent_window.inner_size();
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

        let qfi_if_needed = [graphics_queue.qfi, present_queue.qfi];

        let shared_graphics_present_queue = graphics_queue.qfi == present_queue.qfi;

        let swapchain_ci = SwapchainCreateInfoKHR {
            surface: surface.inner,
            min_image_count: swap_image_count,
            image_format: swapchain_format.format,
            image_color_space: swapchain_format.color_space,
            image_extent: swap_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: if shared_graphics_present_queue {
                SharingMode::EXCLUSIVE
            } else {
                SharingMode::CONCURRENT
            },
            queue_family_index_count: if shared_graphics_present_queue { 0 } else { 2 },
            p_queue_family_indices: if shared_graphics_present_queue {
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
        unsafe { swapchain_device.create_swapchain(&swapchain_ci, ALLOCATION_CALLBACKS.as_ref()) }
            .map(|swapchain| Swapchain {
                inner: swapchain,
                _parent: device.clone(),
                device: swapchain_device,
                _graphics_queue: graphics_queue,
                _present_queue: present_queue,
            })
            .map_err(|_| Error::SwapchainCreation)
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_swapchain(self.inner, ALLOCATION_CALLBACKS.as_ref())
        };
    }
}

impl DebugMessenger {
    fn new(instance: &Arc<Instance>, validation_level: ValidationLevel) -> Option<Self> {
        let debug_instance = debug_utils::Instance::new(&instance.parent, &instance.inner);
        let create_info = DebugUtilsMessengerCreateInfoEXT {
            message_severity: graphics_validation_sev_to_debug_utils_flags(validation_level),
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
                Some(Self {
                    inner: messenger,
                    _parent: instance.clone(),
                    instance: debug_instance,
                })
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
    }
}
impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(win: Arc<Window>, mut opts: ContextCreateOpts) -> Result<Self, Error> {
        //SAFETY: You may not call vulkan functions after Entry is dropped.
        //Therefore Entry should be the last thing dropped.
        let entry = Arc::new(unsafe { Entry::load().map_err(Error::Loading) }?);

        let required_instance_extensions =
            ash_window::enumerate_required_extensions(win.display_handle().unwrap().as_raw())
                .unwrap();

        let required_instance_extensions = required_instance_extensions
            .iter()
            .map(|cstr| unsafe { CStr::from_ptr(*cstr) }.to_str().unwrap())
            .collect::<Vec<_>>();

        let instance = Arc::new(Instance::new(
            entry.clone(),
            &required_instance_extensions,
            &mut opts,
        )?);

        let logger = if let ValidationLevel::None = opts.graphics_validation_layers {
            None
        } else {
            DebugMessenger::new(&instance, opts.graphics_validation_layers)
        };

        let surface = Arc::new(Surface::new(&instance, &win)?);

        let required_device_extensions = [khr::swapchain::NAME.to_str().to_owned().unwrap()];

        let mut missing_instance_extensions = Vec::new();

        let phys_devs = instance.enumerate_physical_devices();

        let (_, phys_dev, qfi) = phys_devs
            .iter()
            .map(|phys_dev| {
                let (score, qfp) =
                    evaluate_phys_dev(phys_dev, &instance, &surface, |extension_list| {
                        for ext in required_device_extensions {
                            //SAFETY: Lifetime of this cstr is less than
                            //lifetime of ext
                            if !extension_list.iter().any(|dev_ext| {
                                dev_ext
                                    .extension_name_as_c_str()
                                    .unwrap()
                                    .to_str()
                                    .unwrap()
                                    .eq(ext)
                            }) {
                                missing_instance_extensions.push(ext)
                            }
                        }
                        if missing_instance_extensions.is_empty() {
                            Some(0)
                        } else {
                            None
                        }
                    });
                (score, phys_dev, qfp)
            })
            .rev()
            .max_by(|(score, _, _), (score2, _, _)| score.cmp(score2))
            .filter(|(score, _, _)| *score > 0)
            .ok_or(Error::NoSuitableDevice)?;
        let qfi = qfi.unwrap();

        let dev = Arc::new(
            instance
                .create_device(phys_dev, &qfi, &required_device_extensions[..])
                .map_err(|_| Error::DeviceCreation)?,
        );

        let graphics_queue = dev.get_device_queue(qfi.graphics, 0);

        let present_queue = dev.get_device_queue(qfi.present, 0);

        let swapchain = Swapchain::new(
            dev.clone(),
            phys_dev,
            surface.clone(),
            graphics_queue,
            present_queue,
        )?;

        let graphics_context = Context {
            _win: win,
            _nd: ContextNonDebug {
                _swapchain: swapchain,
                _logger: logger,
                _entry: entry,
                _instance: instance,
                _dev: dev,
                _surface: surface,
            },
        };
        Ok(graphics_context)
    }
}

fn graphics_validation_sev_to_debug_utils_flags(
    graphics_validation_layers: ValidationLevel,
) -> DebugUtilsMessageSeverityFlagsEXT {
    let none = DebugUtilsMessageSeverityFlagsEXT::empty();
    let error = none | DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let warning = error | DebugUtilsMessageSeverityFlagsEXT::WARNING;
    let info = warning | DebugUtilsMessageSeverityFlagsEXT::INFO;
    let verbose = info | DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
    match graphics_validation_layers {
        ValidationLevel::None => none,
        ValidationLevel::Error => error,
        ValidationLevel::Warning => warning,
        ValidationLevel::Info => info,
        ValidationLevel::Verbose => verbose,
    }
}

struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

#[derive(Clone, Copy)]
struct QueueFamilyIndicesOpt {
    graphics: Option<u32>,
    present: Option<u32>,
}

impl QueueFamilyIndicesOpt {
    fn resolve(&self) -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics: self.graphics.unwrap(),
            present: self.present.unwrap(),
        }
    }

    fn find(phys_dev: &PhysicalDevice, instance: &Instance, surface: &Surface) -> Self {
        //SAFETY: Dropped before instance is destroyed
        let queue_families = instance.get_physical_device_queue_family_properties(phys_dev);
        let candidate_graphics_queues = queue_families
            .iter()
            .enumerate()
            .map(|(i, qfp)| (i as u32, qfp))
            .filter(|(_, qfp)| (qfp.queue_flags.intersects(vk::QueueFlags::GRAPHICS)))
            .collect::<Vec<_>>();
        let graphics = candidate_graphics_queues
            .iter()
            //SAFETY: Due to the properties of enumerate, i is always in bounds
            .filter(|(i, _)| surface.get_physical_device_surface_support(phys_dev, *i))
            .map(|(i, _)| i)
            .next()
            .or_else(|| candidate_graphics_queues.iter().map(|(i, _)| i).next())
            .copied();

        let present = graphics
            .iter()
            //SAFETY: graphics queue index is always in bounds
            .filter(|graphics_queue_index| {
                surface.get_physical_device_surface_support(phys_dev, **graphics_queue_index)
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

                        surface.get_physical_device_surface_support(phys_dev, *queue_family_index)
                    })
            });

        Self { graphics, present }
    }
}

fn evaluate_phys_dev<F: FnOnce(&[vk::ExtensionProperties]) -> Option<u32>>(
    phys_dev: &PhysicalDevice,
    instance: &Instance,
    surface: &Surface,
    score_extensions: F,
) -> (u32, Option<QueueFamilyIndices>) {
    match score_extensions(&instance.enumerate_device_extension_properties(phys_dev)) {
        Some(mut score) => {
            //SAFETY: We discard features before instance is destroyed
            let features = instance.get_physical_device_features(phys_dev);
            //SAFETY: We discard features before instance is destroyed
            let props = instance.get_physical_device_properties(phys_dev);

            score += match props.device_type {
                PhysicalDeviceType::DISCRETE_GPU => 100,
                PhysicalDeviceType::INTEGRATED_GPU => 50,
                PhysicalDeviceType::VIRTUAL_GPU => 25,
                _ => 10,
            };

            let mut suitable = true;

            let queue_family_indexes = QueueFamilyIndicesOpt::find(phys_dev, instance, surface);

            suitable &= features.geometry_shader != 0
                && queue_family_indexes.graphics.is_some()
                && queue_family_indexes.present.is_some();

            if suitable {
                //TODO: Better grading
                (score, Some(queue_family_indexes.resolve()))
            } else {
                (0, None)
            }
        }
        None => (0, None),
    }
}
