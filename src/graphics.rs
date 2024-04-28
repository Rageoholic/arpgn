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
    vk::{self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT},
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
}

struct Surface {
    inner: vk::SurfaceKHR,
    instance: surface::Instance,
    _parent_instance: Arc<Instance>,
    parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        //SAFETY: Last use of surface, all child objects are destroyed
        unsafe { self.instance.destroy_surface(self.inner, None) }
    }
}

struct PhysicalDevice {
    inner: vk::PhysicalDevice,
    _parent: Arc<Instance>,
}

const VERT_SHADER_SOURCE: &str = include_str!("shader.vert");
const FRAG_SHADER_SOURCE: &str = include_str!("shader.frag");

impl Surface {
    fn get_physical_device_surface_support(&self, phys_dev: &PhysicalDevice, i: u32) -> bool {
        //SAFETY: fetching owned data from a valid phys_dev
        unsafe {
            self.instance
                .get_physical_device_surface_support(phys_dev.inner, i, self.inner)
                .unwrap()
        }
    }

    fn get_physical_device_surface_capabilities(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> vk::SurfaceCapabilitiesKHR {
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
    ) -> Vec<vk::PresentModeKHR> {
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
        //SAFETY: Valid window handles, valid instance, Surface destroyed only
        //after all subobjects
        let surface = unsafe {
            ash_window::create_surface(
                &instance.parent,
                &instance.inner,
                win.display_handle().unwrap().as_raw(),
                win.window_handle().unwrap().as_raw(),
                None,
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
    logger: Option<(vk::DebugUtilsMessengerEXT, debug_utils::Instance)>,
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
    fn create_shader_module(
        self: &Arc<Self>,
        source: shaderc::CompilationArtifact,
    ) -> ShaderModule {
        let shader_ci = vk::ShaderModuleCreateInfo::default().code(source.as_binary());
        //SAFETY: Valid SPIR-V, these are tied together via Drop
        let shader_module = unsafe { self.inner.create_shader_module(&shader_ci, None).unwrap() };
        ShaderModule {
            inner: shader_module,
            parent: self.clone(),
        }
    }

    fn get_device_queue(self: &Arc<Self>, qfi: u32, queue_index: u32) -> Queue {
        Queue {
            //SAFETY: This must be dropped in order to drop the device
            _inner: unsafe { self.inner.get_device_queue(qfi, queue_index) },
            _parent: self.clone(),
            qfi,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //SAFETY: Last use of device. All other references should have been dropped
        unsafe { self.inner.destroy_device(None) };
    }
}

impl Instance {
    fn get_physical_device_features(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> vk::PhysicalDeviceFeatures {
        //SAFETY: Getting owned data with valid phys_dev
        unsafe { self.inner.get_physical_device_features(phys_dev.inner) }
    }

    fn new(
        entry: Arc<Entry>,

        required_instance_extensions: &[&str],
        mut opts: ContextCreateOpts,
    ) -> Result<Self, Error> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .application_name(c"placeholder")
            .engine_name(c"placeholder");

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

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layers);

        //SAFETY: cannot be used after entry is dropped. All pointers in the
        //create infos and associated structs must be valid. allocation
        //callbacks must be None or valid
        //
        //Allocation callbacks:
        //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|_| Error::InstanceCreation)?;

        let logger = if opts.graphics_validation_layers != ValidationLevel::None {
            let debug_instance = debug_utils::Instance::new(&entry, &instance);
            let debug_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(graphics_validation_sev_to_debug_utils_flags(
                    opts.graphics_validation_layers,
                ))
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(debug_callback));

            let debug_messenger = {
                //SAFETY: Should always be fine if we got here. We added the layers
                //we care about and the extension
                unsafe { debug_instance.create_debug_utils_messenger(&debug_ci, None) }.unwrap()
            };

            let debug_message_info = vk::DebugUtilsMessengerCallbackDataEXT::default()
                .message(c"Test message")
                .message_id_name(c"test");

            //SAFETY: Should always be fine
            unsafe {
                debug_instance.submit_debug_utils_message(
                    DebugUtilsMessageSeverityFlagsEXT::INFO,
                    DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    &debug_message_info,
                )
            };

            Some((debug_messenger, debug_instance))
        } else {
            None
        };
        Ok(Instance {
            inner: instance,
            parent: entry.clone(),
            logger,
        })
    }

    fn get_physical_device_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        //SAFETY: Phys dev guarantees valid
        unsafe { self.inner.get_physical_device_properties(phys_dev.inner) }
    }

    fn get_physical_device_queue_family_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        //SAFETY: PhysicalDevice guarantess inner is valid
        unsafe {
            self.inner
                .get_physical_device_queue_family_properties(phys_dev.inner)
        }
    }

    fn enumerate_device_extension_properties(
        &self,
        phys_dev: &PhysicalDevice,
    ) -> Vec<vk::ExtensionProperties> {
        //SAFETY: PhysicalDevice guarantees inner is valid
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
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_priorities(&queue_priorities)
                    .queue_family_index(queue_family_index)
            })
            .collect::<Vec<_>>();

        let dev_features = vk::PhysicalDeviceFeatures::default();

        let required_device_extensions = required_device_extensions
            .iter()
            .map(|ext_name| CString::new(&ext_name[..]).unwrap())
            .collect::<Vec<_>>();

        let required_device_extensions = required_device_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let dev_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&dev_queue_cis)
            .enabled_features(&dev_features)
            .enabled_extension_names(&required_device_extensions);

        //SAFETY: Device is tied to an rc pointer to the parent
        unsafe {
            self.inner
                .create_device(phys_dev.inner, &dev_ci, None)
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

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some((ref debug_messenger, ref debug_instance)) = self.logger {
            //SAFETY: Should always be fine
            unsafe { debug_instance.destroy_debug_utils_messenger(*debug_messenger, None) }
        }

        //SAFETY: This is always valid
        unsafe { self.inner.destroy_instance(None) };
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
    let log_id_cstr = unsafe { CStr::from_ptr(callback_data.p_message_id_name) };

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
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "graphics_subsystem.debug_utils.general",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "graphics_subsystem.debug_utils.perf",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => {
            "graphics_subsystem.debug_utils.validation"
        }
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => {
            "graphics_subsystem.debug_utils.device_address_binding"
        }
        _ => "graphics_subsystem.debug_utils.unknown",
    }
}

const KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

struct Swapchain {
    inner: vk::SwapchainKHR,
    parent: Arc<Device>,
    device: swapchain::Device,
    _graphics_queue: Queue,
    _present_queue: Queue,
    image_views: Vec<vk::ImageView>,
    _format: vk::Format,
    _images: Vec<vk::Image>,
    _extent: vk::Extent2D,
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
            .find(|present_mode| *present_mode == vk::PresentModeKHR::MAILBOX)
            .or_else(|| {
                swapchain_present_modes
                    .iter()
                    .copied()
                    .find(|present_mode| *present_mode == vk::PresentModeKHR::IMMEDIATE)
            })
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let win_size = surface.parent_window.inner_size();
        let swap_extent = vk::Extent2D {
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

        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.inner,
            min_image_count: swap_image_count,
            image_format: swapchain_format.format,
            image_color_space: swapchain_format.color_space,
            image_extent: swap_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: if shared_graphics_present_queue {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            queue_family_index_count: if shared_graphics_present_queue { 0 } else { 2 },
            p_queue_family_indices: if shared_graphics_present_queue {
                null()
            } else {
                qfi_if_needed.as_ptr()
            },
            pre_transform: swapchain_capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: swapchain_present_mode,
            clipped: vk::TRUE,

            ..Default::default()
        };

        //SAFETY: Allocation callbacks must be valid, swapchain_ci must be
        //valid, swapchain must be destroyed before surface
        let swapchain = unsafe { swapchain_device.create_swapchain(&swapchain_ci, None) }
            .map_err(|_| Error::SwapchainCreation)?;

        //SAFETY: Images are implicitly destroyed with the swapchain, tied together via Drop
        let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }.unwrap();
        let image_views = images
            .iter()
            .map(|image| {
                let ivci = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                //SAFETY: Image view is destroyed before Image
                unsafe { device.inner.create_image_view(&ivci, None) }.unwrap()
            })
            .collect();

        Ok(Swapchain {
            inner: swapchain,
            parent: device.clone(),
            device: swapchain_device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            _extent: swap_extent,
            _images: images,
            _format: swapchain_format.format,
            image_views,
        })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        //SAFETY: These do not escape Swapchain. It owns them wholly
        unsafe {
            for image in self.image_views.iter() {
                self.parent.inner.destroy_image_view(*image, None)
            }
            self.device.destroy_swapchain(self.inner, None)
        };
    }
}

impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(win: Arc<Window>, opts: ContextCreateOpts) -> Result<Self, Error> {
        //SAFETY: You may not call vulkan functions after Entry is dropped.
        //Therefore Entry should be the last thing dropped.
        let entry = Arc::new(unsafe { Entry::load().map_err(Error::Loading) }?);

        let required_instance_extensions =
            ash_window::enumerate_required_extensions(win.display_handle().unwrap().as_raw())
                .unwrap();

        let required_instance_extensions = required_instance_extensions
            .iter()
            //SAFETY: We know these are fine
            .map(|cstr| unsafe { CStr::from_ptr(*cstr) }.to_str().unwrap())
            .collect::<Vec<_>>();

        let instance = Arc::new(Instance::new(
            entry.clone(),
            &required_instance_extensions,
            opts,
        )?);

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

        let shader_compiler = shaderc::Compiler::new().unwrap();
        let vert_shader_spirv = shader_compiler
            .compile_into_spirv(
                VERT_SHADER_SOURCE,
                shaderc::ShaderKind::Vertex,
                "shader.vert",
                "main",
                None,
            )
            .unwrap();

        let frag_shader_spirv = shader_compiler
            .compile_into_spirv(
                FRAG_SHADER_SOURCE,
                shaderc::ShaderKind::Fragment,
                "shader.frag",
                "main",
                None,
            )
            .unwrap();

        let _vert_shader_mod = dev.create_shader_module(vert_shader_spirv);
        let _frag_shader_mod = dev.create_shader_module(frag_shader_spirv);

        let graphics_context = Context {
            _win: win,
            _nd: ContextNonDebug {
                _swapchain: swapchain,
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
                vk::PhysicalDeviceType::DISCRETE_GPU => 100,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 25,
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

struct ShaderModule {
    inner: vk::ShaderModule,
    parent: Arc<Device>,
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        //SAFETY: This should just always be fine so long as we're passing the
        //correct allocation callbacks?
        unsafe { self.parent.inner.destroy_shader_module(self.inner, None) };
    }
}
