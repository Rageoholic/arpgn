use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    fmt::Debug,
    marker::PhantomData,
    mem::{size_of, ManuallyDrop},
    os::raw::c_void,
    ptr::null,
    sync::{Arc, RwLock},
};

use ash::{
    ext::debug_utils,
    khr::{self, surface, swapchain},
    prelude::VkResult,
    vk::{self},
    Entry, LoadingError,
};

use structopt::StructOpt;
use strum::EnumString;
use vek::{Vec2, Vec3};
use vk_mem::Alloc;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    pos: Vec2<f32>,
    col: Vec3<f32>,
}

const VERTICES: [Vertex; 3] = [
    Vertex::new(Vec2::new(0.0, -0.5), Vec3::new(1.0, 0.0, 0.0)),
    Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(0.0, 1.0, 0.0)),
    Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(0.0, 0.0, 1.0)),
];

impl Vertex {
    const fn new(pos: Vec2<f32>, col: Vec3<f32>) -> Self {
        Self { pos, col }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_description() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(Self, pos) as u32);
        let col = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(Self, col) as u32);
        [pos, col]
    }
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct ContextNonDebug {
    _entry: Arc<Entry>,
    _instance: Arc<Instance>,
    render_data: RenderData,
    _dev: Arc<Device>,
    _surface: Arc<Surface>,
    pipeline_layout: PipelineLayout,
    shader_stages: ShaderStages,
}

struct Surface {
    inner: RwLock<vk::SurfaceKHR>,
    instance: surface::Instance,
    parent_instance: Arc<Instance>,
    parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        //SAFETY: Last use of surface, all child objects are destroyed
        unsafe {
            self.instance
                .destroy_surface(*self.inner.get_mut().unwrap(), None)
        }
    }
}

const VERT_SHADER_SOURCE: &str = include_str!("shader.vert");
const FRAG_SHADER_SOURCE: &str = include_str!("shader.frag");

impl Surface {
    unsafe fn get_physical_device_surface_support(
        &self,
        phys_dev: vk::PhysicalDevice,
        i: u32,
    ) -> bool {
        //SAFETY: fetching owned data from a valid phys_dev
        unsafe {
            self.instance
                .get_physical_device_surface_support(phys_dev, i, *self.inner.read().unwrap())
                .unwrap()
        }
    }

    unsafe fn get_physical_device_surface_capabilities(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> vk::SurfaceCapabilitiesKHR {
        //SAFETY: Copying owned data
        unsafe {
            self.instance
                .get_physical_device_surface_capabilities(phys_dev, *self.inner.read().unwrap())
        }
        .unwrap()
    }

    unsafe fn get_physical_device_surface_present_modes(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> Vec<vk::PresentModeKHR> {
        //SAFETY: fetching owned data
        unsafe {
            self.instance
                .get_physical_device_surface_present_modes(phys_dev, *self.inner.read().unwrap())
        }
        .unwrap()
    }
    unsafe fn get_physical_device_surface_formats(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        //SAFETY: This is okay because surface is known to be valid and we're
        //getting a Vec of PODs
        unsafe {
            self.instance
                .get_physical_device_surface_formats(phys_dev, *self.inner.read().unwrap())
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
            inner: RwLock::new(surface),
            instance: surface_instance,
            parent_instance: instance.clone(),
            parent_window: win.clone(),
        })
    }
}

struct PipelineLayout {
    inner: vk::PipelineLayout,
    parent: Arc<Device>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        //SAFETY: We know this is okay because our safety requirement is that
        //this is dropped after any child object and we make sure of that
        unsafe { self.parent.inner.destroy_pipeline_layout(self.inner, None) };
    }
}

struct Instance {
    inner: ash::Instance,
    parent: Arc<Entry>,
    logger: Option<(vk::DebugUtilsMessengerEXT, debug_utils::Instance)>,
}

struct Queue {
    qfi: u32,
    inner: RwLock<vk::Queue>,
    _parent: Arc<Device>,
}

struct Device {
    inner: ash::Device,
    parent: Arc<Instance>,
    phys_dev: vk::PhysicalDevice,
    allocator: ManuallyDrop<vk_mem::Allocator>,
    memory_type_info: vk::PhysicalDeviceMemoryProperties,
}

struct GraphicsPipeline {
    inner: vk::Pipeline,
    parent: Arc<Device>,
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        //SAFETY: We made the pipeline in here from parent
        unsafe { self.parent.inner.destroy_pipeline(self.inner, None) }
    }
}

struct CommandBuffer {
    inner: vk::CommandBuffer,
    parent: Arc<CommandPool>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        let command_buffers = [self.inner];
        //SAFETY: Synchronize access to command_pool via rwlock. Exclusively own
        //command_buffers. command buffers is derived from command_pool
        unsafe {
            self.parent
                .parent
                .inner
                .free_command_buffers(*self.parent.inner.write().unwrap(), &command_buffers)
        }
    }
}

struct CommandPool {
    inner: RwLock<vk::CommandPool>,
    parent: Arc<Device>,
}

impl CommandPool {
    fn create_command_buffers(self: &Arc<Self>, count: u32) -> VkResult<Vec<CommandBuffer>> {
        let inner = self.inner.write().unwrap();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(count)
            .command_pool(*inner);

        //SAFETY: alloc_info is valid. Inner is an exclusive ref to the command pool
        unsafe { self.parent.inner.allocate_command_buffers(&alloc_info) }.map(|command_buffers| {
            command_buffers
                .iter()
                .copied()
                .map(|command_buffer| CommandBuffer {
                    inner: command_buffer,
                    parent: self.clone(),
                })
                .collect()
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        //SAFETY: We own these values and anyone holding onto a command pool from here needs to hold a ref to this.
        //Additionally, we have an &mut ref which means we have exclusive ownership for any synch requirements
        unsafe {
            self.parent
                .inner
                .destroy_command_pool(*self.inner.get_mut().unwrap(), None)
        }
    }
}

impl Device {
    fn get_inner(&self) -> &ash::Device {
        &self.inner
    }
    unsafe fn create_command_pool(
        self: &Arc<Self>,
        create_info: &vk::CommandPoolCreateInfo,
    ) -> VkResult<CommandPool> {
        //SAFETY: create_info is valid
        unsafe { self.inner.create_command_pool(create_info, None) }.map(|inner| CommandPool {
            inner: RwLock::new(inner),
            parent: self.clone(),
        })
    }
    unsafe fn create_graphics_pipelines(
        self: &Arc<Self>,
        create_infos: &[vk::GraphicsPipelineCreateInfo],
    ) -> VkResult<Vec<GraphicsPipeline>> {
        //SAFETY: Using transient safety from being in an unsafe fn, create_info is valid
        unsafe {
            self.inner
                .create_graphics_pipelines(vk::PipelineCache::null(), create_infos, None)
        }
        .map(|v| {
            v.iter()
                .map(|inner| GraphicsPipeline {
                    inner: *inner,
                    parent: self.clone(),
                })
                .collect()
        })
        .map_err(|(_, err)| err)
    }

    unsafe fn create_render_pass(
        self: &Arc<Self>,
        create_info: &vk::RenderPassCreateInfo,
    ) -> VkResult<RenderPass> {
        //SAFETY: Using transient safety from being in an unsafe fn, create_info is valid
        unsafe { self.inner.create_render_pass(create_info, None) }.map(|inner| RenderPass {
            inner,
            parent: self.clone(),
        })
    }
    unsafe fn create_pipeline_layout(
        self: &Arc<Self>,
        create_info: &vk::PipelineLayoutCreateInfo,
    ) -> VkResult<PipelineLayout> {
        //SAFETY: Using transient safety from being in an unsafe fn, create_info is valid
        unsafe { self.inner.create_pipeline_layout(create_info, None) }.map(|inner| {
            PipelineLayout {
                inner,
                parent: self.clone(),
            }
        })
    }

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
            inner: RwLock::new(unsafe { self.inner.get_device_queue(qfi, queue_index) }),
            _parent: self.clone(),
            qfi,
        }
    }
}

struct ShaderStages {
    vert: ShaderModule,
    frag: ShaderModule,
}

impl ShaderStages {
    //SAFETY: This array has an implicitly tied lifetime to self
    unsafe fn to_array(&self) -> Vec<vk::PipelineShaderStageCreateInfo> {
        vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(self.vert.inner)
                .stage(vk::ShaderStageFlags::VERTEX)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .module(self.frag.inner)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .name(c"main"),
        ]
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //SAFETY: Last use of device. All other references should have been
        //dropped. Last use of self.allocator
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
            self.inner.destroy_device(None);
        }
    }
}

impl Instance {
    unsafe fn get_physical_device_features(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceFeatures {
        //SAFETY: Getting owned data with valid phys_dev
        unsafe { self.inner.get_physical_device_features(phys_dev) }
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

        //SAFETY: We know this is safe because we drop this Vec before we drop
        //entry
        let avail_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|_| Error::InstanceCreation)?;

        //SAFETY: We know this is safe because we drop this Vec before we drop entry
        let avail_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|_| Error::InstanceCreation)?;

        let mut missing_instance_extensions = Vec::new();

        for ext in required_instance_extensions.iter().copied() {
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
                //SAFETY: Should always be fine if we got here. We added the
                //layers we care about and the extension
                unsafe { debug_instance.create_debug_utils_messenger(&debug_ci, None) }.unwrap()
            };

            let debug_message_info = vk::DebugUtilsMessengerCallbackDataEXT::default()
                .message(c"Test message")
                .message_id_name(c"test");

            //SAFETY: Should always be fine
            unsafe {
                debug_instance.submit_debug_utils_message(
                    vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
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

    unsafe fn get_physical_device_properties(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        //SAFETY: Phys dev guarantees valid
        unsafe { self.inner.get_physical_device_properties(phys_dev) }
    }

    unsafe fn get_physical_device_queue_family_properties(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        //SAFETY: PhysicalDevice guarantess inner is valid
        unsafe {
            self.inner
                .get_physical_device_queue_family_properties(phys_dev)
        }
    }

    unsafe fn enumerate_device_extension_properties(
        &self,
        phys_dev: vk::PhysicalDevice,
    ) -> Vec<vk::ExtensionProperties> {
        //SAFETY: PhysicalDevice guarantees inner is valid
        unsafe { self.inner.enumerate_device_extension_properties(phys_dev) }.unwrap()
    }
    unsafe fn create_device(
        self: &Arc<Self>,
        phys_dev: vk::PhysicalDevice,
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
                .create_device(phys_dev, &dev_ci, None)
                .map(|d| {
                    let allocator = ManuallyDrop::new(
                        vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
                            &self.inner,
                            &d,
                            phys_dev,
                        ))
                        .unwrap(),
                    );
                    let memory_type_info =
                        self.inner.get_physical_device_memory_properties(phys_dev);
                    Device {
                        inner: d,
                        parent: self.clone(),
                        phys_dev,
                        allocator,
                        memory_type_info,
                    }
                })
                .map_err(|_| Error::DeviceCreation)
        }
    }

    fn enumerate_physical_devices(self: &Arc<Self>) -> Vec<vk::PhysicalDevice> {
        //SAFETY: this vec is dropped before instance is destroyed
        unsafe { self.inner.enumerate_physical_devices() }.unwrap()
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
    nd: ContextNonDebug,
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
    CommandBufferCreation,
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

struct Replaceable {
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    inner: vk::SwapchainKHR,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: RenderPass,
    graphics_pipeline: GraphicsPipeline,
    device: Arc<Device>,
    swapchain_device: Arc<swapchain::Device>,
}

impl Replaceable {
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        swapchain_device: &Arc<swapchain::Device>,
        surface: &Arc<Surface>,
        graphics_queue: &Queue,
        present_queue: &Queue,
        device: &Arc<Device>,
        shader_stages: &ShaderStages,
        pipeline_layout: &PipelineLayout,
        old_swapchain: Option<&vk::SwapchainKHR>,
    ) -> VkResult<Self> {
        let win_size = surface.parent_window.inner_size();
        if win_size.width != 0 && win_size.height != 0 {
            //SAFETY: Phys dev is derived from same instance
            let (swapchain_formats, swapchain_present_modes, swapchain_capabilities) = unsafe {
                let swapchain_formats = surface
                    .get_physical_device_surface_formats(device.phys_dev)
                    .unwrap();

                let swapchain_present_modes =
                    surface.get_physical_device_surface_present_modes(device.phys_dev);

                let swapchain_capabilities =
                    surface.get_physical_device_surface_capabilities(device.phys_dev);
                (
                    swapchain_formats,
                    swapchain_present_modes,
                    swapchain_capabilities,
                )
            };
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

            //create_swapchain_KHR requires external sync on the surface
            let surface_mut = surface.inner.write().unwrap();

            let swapchain_ci = vk::SwapchainCreateInfoKHR {
                surface: *surface_mut,
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
                old_swapchain: old_swapchain.copied().unwrap_or(vk::SwapchainKHR::null()),
                ..Default::default()
            };
            //SAFETY: Allocation callbacks must be valid, swapchain_ci must be
            //valid, swapchain must be destroyed before surface
            let swapchain = unsafe { swapchain_device.create_swapchain(&swapchain_ci, None) }?;

            //SAFETY: Images are implicitly destroyed with the swapchain, tied together via Drop
            let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }.unwrap();
            let image_views: Vec<_> = images
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

            let color_attachment_refs = [vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_attachments = [vk::AttachmentDescription::default()
                .format(swapchain_ci.image_format)
                .samples(multisampling.rasterization_samples)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_refs)];

            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: Default::default(),
            }];

            let render_pass_ci = vk::RenderPassCreateInfo::default()
                .attachments(&color_attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);

            let pipeline_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
            let viewports = [vk::Viewport::default()
                .width(swap_extent.width as f32)
                .height(swap_extent.height as f32)];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swap_extent,
            }];

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);

            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);

            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];

            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_attachments);

            //SAFETY: lifetimes are tied here
            let shader_stages_cis = unsafe { shader_stages.to_array() };

            let vertex_binding_descriptions = [Vertex::binding_description()];
            let vertex_attrribute_descriptions = Vertex::attribute_description();

            let pipeline_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descriptions[..])
                .vertex_attribute_descriptions(&vertex_attrribute_descriptions[..]);

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let pipeline_dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let render_pass =
            //SAFETY: render_pass_ci must be valid, we create it
             unsafe { device.create_render_pass(&render_pass_ci) }.unwrap();
            let framebuffers: Vec<_> = image_views
                .iter()
                .map(|iv| {
                    let attachments = [*iv];
                    let framebuffer_ci = vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass.inner)
                        .layers(1)
                        .attachments(&attachments)
                        .width(swap_extent.width)
                        .height(swap_extent.height);

                    //SAFETY: valid ci
                    unsafe {
                        device
                            .inner
                            .create_framebuffer(&framebuffer_ci, None)
                            .unwrap()
                    }
                })
                .collect();

            let graphics_pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages_cis)
                .layout(pipeline_layout.inner)
                .vertex_input_state(&pipeline_vertex_input)
                .input_assembly_state(&pipeline_input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .color_blend_state(&color_blending)
                .dynamic_state(&pipeline_dynamic_state)
                .layout(pipeline_layout.inner)
                .render_pass(render_pass.inner)
                .subpass(0);

            let graphics_pipeline =
            //SAFETY: CI must be valid
            unsafe { device.create_graphics_pipelines(&[graphics_pipeline_ci]) }
                .unwrap()
                .pop()
                .unwrap();

            Ok(Self {
                extent: swap_extent,
                image_views,
                inner: swapchain,
                framebuffers,
                render_pass,
                graphics_pipeline,
                device: device.clone(),
                swapchain_device: swapchain_device.clone(),
            })
        } else {
            Err(vk::Result::SUCCESS)
        }
    }
}

struct Buffer<T> {
    inner: vk::Buffer,
    parent: Arc<Device>,
    allocation: vk_mem::Allocation,
    _size: u64,
    _phantom: PhantomData<T>,
    _allocation_info: vk_mem::AllocationInfo,
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        //SAFETY: Allocator allocated this buffer. Buffer and allocation are
        //correctly tied together
        unsafe {
            self.parent
                .allocator
                .destroy_buffer(self.inner, &mut self.allocation)
        };
    }
}

struct RenderData {
    parent: Arc<Device>,
    graphics_queue: Queue,
    present_queue: Queue,
    command_buffers: Vec<CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    main_vertex_buffers: Vec<Buffer<Vertex>>,
    frame: usize,
    surface: Arc<Surface>,
    swapchain_device: Arc<swapchain::Device>,
    r: Option<Replaceable>,
}

impl RenderData {
    //TODO: Clean this up

    fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
        graphics_queue: Queue,
        present_queue: Queue,
        command_pool: Arc<CommandPool>,
        pipeline_layout: &PipelineLayout,
        shader_stages: &ShaderStages,
    ) -> Result<Self, Error> {
        let swapchain_device =
            Arc::new(swapchain::Device::new(&device.parent.inner, &device.inner));

        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(
                //SAFETY: We destroy this in Swapchain::drop() before device is destroyed
                unsafe { device.inner.create_semaphore(&semaphore_ci, None) }.unwrap(),
            );
            render_finished_semaphores.push(
                //SAFETY: Destroyed in Swapchain::drop()
                unsafe { device.inner.create_semaphore(&semaphore_ci, None) }.unwrap(),
            );
            in_flight_fences.push(
                //SAFETY: Destroyed in Swapchain::drop()
                unsafe { device.inner.create_fence(&fence_ci, None) }.unwrap(),
            );
        }

        let command_buffers = command_pool
            .create_command_buffers(MAX_FRAMES_IN_FLIGHT as u32)
            .map_err(|_| Error::CommandBufferCreation)?;

        let vertex_buffer_ci = vk::BufferCreateInfo::default()
            .size(size_of::<Vertex>() as u64 * 3)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER);
        let vertex_buffer_allocation_ci = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vk_mem::MemoryUsage::AutoPreferDevice,

            ..Default::default()
        };

        let mut main_vertex_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            //SAFETY: Valid buffer create info and allocation infos
            let (buffer, mut allocation) = unsafe {
                device
                    .allocator
                    .create_buffer(&vertex_buffer_ci, &vertex_buffer_allocation_ci)
            }
            .unwrap();
            let allocation_info = device.allocator.get_allocation_info(&allocation);

            //SAFETY: Valid allocation
            unsafe {
                let mapping = device.allocator.map_memory(&mut allocation).unwrap();
                std::ptr::copy_nonoverlapping(
                    VERTICES.as_ptr() as *const u8,
                    mapping,
                    std::mem::size_of_val(&VERTICES),
                );

                if device.memory_type_info.memory_types_as_slice()
                    [allocation_info.memory_type as usize]
                    .property_flags
                    & vk::MemoryPropertyFlags::HOST_COHERENT
                    == vk::MemoryPropertyFlags::empty()
                {
                    device
                        .allocator
                        .flush_allocation(&allocation, 0, vk::WHOLE_SIZE)
                        .unwrap();
                }
                device.allocator.unmap_memory(&mut allocation);
            }

            main_vertex_buffers.push(Buffer {
                inner: buffer,
                allocation,
                _size: vertex_buffer_ci.size / size_of::<Vertex>() as u64,
                parent: device.clone(),
                _allocation_info: allocation_info,
                _phantom: PhantomData,
            });
        }

        //SAFETY: Known to be good
        let r = unsafe {
            Replaceable::new(
                &swapchain_device,
                &surface,
                &graphics_queue,
                &present_queue,
                &device,
                shader_stages,
                pipeline_layout,
                None,
            )
            .map_err(|_| Error::SwapchainCreation)
        }?;

        Ok(RenderData {
            r: Some(r),
            parent: device.clone(),
            graphics_queue,
            present_queue,
            command_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            surface: surface.clone(),
            frame: 0,
            swapchain_device,
            main_vertex_buffers,
        })
    }

    fn draw(&mut self) -> VkResult<()> {
        let sync_index = self.frame;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let dev = self.parent.get_inner();
        let swapchain_device = &mut self.swapchain_device;
        if let Some(r) = &mut self.r {
            let in_flight_fence = self.in_flight_fences[sync_index];
            let fences = [in_flight_fence];
            let image_available_semaphore = self.image_available_semaphores[sync_index];
            let render_finished_semaphore = self.render_finished_semaphores[sync_index];
            //SAFETY: fences are valid and from device
            let index = unsafe {
                dev.wait_for_fences(&fences, true, u64::MAX)?;
                dev.reset_fences(&fences)?;

                let index = swapchain_device
                    .acquire_next_image(
                        r.inner,
                        u64::MAX,
                        image_available_semaphore,
                        vk::Fence::null(),
                    )?
                    .0;
                dev.reset_fences(&fences)?;
                index
            };
            let cb = self.command_buffers[sync_index].inner;

            //SAFETY: command buffer comes from dev. Is not in use.
            unsafe { dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())? };
            let dev = self.parent.get_inner();
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

            //SAFETY: We know we have exclusive access to the command buffer because it's &mut

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(r.render_pass.inner)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default().x(0).y(0),
                    extent: r.extent,
                })
                .clear_values(&clear_values)
                .framebuffer(r.framebuffers[index as usize]);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: r.extent.width as f32,
                height: r.extent.height as f32,
                min_depth: 0.0,
                max_depth: 0.0,
            };

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: r.extent,
            };

            //SAFETY: Exclusive access to cb. Valid parameters passed in
            unsafe {
                dev.begin_command_buffer(cb, &command_buffer_begin_info)?;
                dev.cmd_bind_vertex_buffers(
                    cb,
                    0,
                    &[self.main_vertex_buffers[sync_index].inner],
                    &[0],
                );
                dev.cmd_begin_render_pass(cb, &render_pass_begin_info, vk::SubpassContents::INLINE);
                dev.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    r.graphics_pipeline.inner,
                );
                dev.cmd_set_viewport(cb, 0, &[viewport]);
                dev.cmd_set_scissor(cb, 0, &[scissor]);

                dev.cmd_draw(cb, 3, 1, 0, 0);

                dev.cmd_end_render_pass(cb);
                dev.end_command_buffer(cb)?;
            };

            let cbs = [cb];

            let wait_semaphores = [image_available_semaphore];
            let signal_semaphores = [render_finished_semaphore];

            let submit_info = [vk::SubmitInfo::default()
                .command_buffers(&cbs)
                .wait_semaphores(&wait_semaphores)
                .signal_semaphores(&signal_semaphores)];

            let graphics_queue = self.graphics_queue.inner.get_mut().unwrap();

            //SAFETY: Exclusive access to queue, all objects in submit_info and the
            //fence is made with dev
            unsafe { dev.queue_submit(*graphics_queue, &submit_info, in_flight_fence) }?;
            let swapchains = [r.inner];
            let image_indices = [index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let present_queue = self.present_queue.inner.get_mut().unwrap();
            //SAFETY: Use RWLock to synchronize access to present queue. All passed
            //stuff is derived from dev
            unsafe { swapchain_device.queue_present(*present_queue, &present_info) }?;

            Ok(())
        } else {
            Ok(())
        }
    }

    fn resize(
        &mut self,
        pipeline_layout: &PipelineLayout,
        shader_stages: &ShaderStages,
    ) -> VkResult<()> {
        if self.r.is_none() {
            //SAFETY: Pretty much always safe
            unsafe { self.parent.inner.device_wait_idle()? };
        }

        //SAFETY: old_swapchain is valid or None, lifetimes are tied
        self.r = match unsafe {
            Replaceable::new(
                &self.swapchain_device,
                &self.surface,
                &self.graphics_queue,
                &self.present_queue,
                &self.parent,
                shader_stages,
                pipeline_layout,
                self.r.as_ref().map(|r| &r.inner),
            )
        } {
            Ok(r) => Ok(Some(r)),
            Err(vk::Result::SUCCESS) => Ok(None),
            Err(res) => Err(res),
        }?;
        Ok(())
    }
}

impl Drop for Replaceable {
    fn drop(&mut self) {
        let dev = &self.device.inner;
        let swapchain_device = &self.swapchain_device;
        //SAFETY: Exclusive access to queues means no more work submitted
        unsafe { dev.device_wait_idle().unwrap() };
        //SAFETY: Own these objects and have mutable refs to them
        unsafe {
            for framebuffer in self.framebuffers.iter().copied() {
                dev.destroy_framebuffer(framebuffer, None);
            }
            for image in self.image_views.iter() {
                dev.destroy_image_view(*image, None)
            }
            swapchain_device.destroy_swapchain(self.inner, None)
        }
    }
}

impl Drop for RenderData {
    fn drop(&mut self) {
        let dev = &self.parent.as_ref().inner;
        //SAFETY: Exclusive access to queues means no more work submitted
        unsafe { dev.device_wait_idle().unwrap() };
        //SAFETY: These do not escape Swapchain. It owns them wholly
        unsafe {
            for sem in self.image_available_semaphores.drain(..) {
                dev.destroy_semaphore(sem, None);
            }
            for sem in self.render_finished_semaphores.drain(..) {
                dev.destroy_semaphore(sem, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                dev.destroy_fence(fence, None);
            }
        };
    }
}

struct RenderPass {
    inner: vk::RenderPass,
    parent: Arc<Device>,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        //SAFETY: We own renderpass and it's derived from device
        unsafe { self.parent.inner.destroy_render_pass(self.inner, None) };
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

        let mut phys_devs = instance.enumerate_physical_devices();

        let (_, phys_dev, qfi) = phys_devs
            .drain(..)
            .map(|phys_dev| {
                //SAFETY: phys_dev and surface derived from instance
                let (score, qfp) = unsafe {
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
                    })
                };
                (score, phys_dev, qfp)
            })
            .rev()
            .max_by(|(score, _, _), (score2, _, _)| score.cmp(score2))
            .filter(|(score, _, _)| *score > 0)
            .ok_or(Error::NoSuitableDevice)?;
        let qfi = qfi.unwrap();
        //SAFETY: phys_dev derived from instance
        let dev = Arc::new(unsafe {
            instance
                .create_device(phys_dev, &qfi, &required_device_extensions[..])
                .map_err(|_| Error::DeviceCreation)
        }?);

        let graphics_queue = dev.get_device_queue(qfi.graphics, 0);

        let present_queue = dev.get_device_queue(qfi.present, 0);

        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue.qfi);
        //SAFETY: Valid CI
        let command_pool = Arc::new(unsafe { dev.create_command_pool(&command_pool_ci) }.unwrap());

        let pipeline_layout_ci = Default::default();

        let pipeline_layout =
            //SAFETY: pipeline_layout_ci is valid. Known because we made it
            unsafe { dev.create_pipeline_layout(&pipeline_layout_ci) }.unwrap();

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

        let vert_shader_mod = dev.create_shader_module(vert_shader_spirv);
        let frag_shader_mod = dev.create_shader_module(frag_shader_spirv);

        let shader_stages = ShaderStages {
            vert: vert_shader_mod,
            frag: frag_shader_mod,
        };

        let render_data = RenderData::new(
            dev.clone(),
            surface.clone(),
            graphics_queue,
            present_queue,
            command_pool,
            &pipeline_layout,
            &shader_stages,
        )?;

        let graphics_context = Context {
            _win: win,
            nd: ContextNonDebug {
                render_data,
                _entry: entry,
                _instance: instance,
                _dev: dev,
                _surface: surface,
                pipeline_layout,
                shader_stages,
            },
        };
        Ok(graphics_context)
    }
    pub fn resize(&mut self) {
        self.nd
            .render_data
            .resize(&self.nd.pipeline_layout, &self.nd.shader_stages)
            .unwrap();
    }
    pub fn draw(&mut self) {
        match self.nd.render_data.draw() {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.resize(),
            Err(vk::Result::SUBOPTIMAL_KHR) => self.resize(),
            Err(err) => {
                panic!("Result error on draw {}", err);
            }
        };
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

    unsafe fn find(phys_dev: vk::PhysicalDevice, surface: &Surface) -> Self {
        let instance = &surface.parent_instance;
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
            .filter(|(i, _)| unsafe { surface.get_physical_device_surface_support(phys_dev, *i) })
            .map(|(i, _)| i)
            .next()
            .or_else(|| candidate_graphics_queues.iter().map(|(i, _)| i).next())
            .copied();

        let present = graphics
            .iter()
            //SAFETY: graphics queue index is always in bounds
            .filter(|graphics_queue_index| unsafe {
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
                        //SAFETY: queue_family_index will always be in bounds
                        //due to enumerate

                        unsafe {
                            surface
                                .get_physical_device_surface_support(phys_dev, *queue_family_index)
                        }
                    })
            });

        Self { graphics, present }
    }
}

unsafe fn evaluate_phys_dev<F: FnOnce(&[vk::ExtensionProperties]) -> Option<u32>>(
    phys_dev: vk::PhysicalDevice,
    instance: &Instance,
    surface: &Surface,
    score_extensions: F,
) -> (u32, Option<QueueFamilyIndices>) {
    //SAFETY: phys_dev derived from instance
    match score_extensions(&unsafe { instance.enumerate_device_extension_properties(phys_dev) }) {
        Some(mut score) => {
            //SAFETY: We discard features before instance is destroyed
            let features = unsafe { instance.get_physical_device_features(phys_dev) };
            //SAFETY: We discard features before instance is destroyed
            let props = unsafe { instance.get_physical_device_properties(phys_dev) };

            score += match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 100,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 25,
                _ => 10,
            };

            let mut suitable = true;

            //SAFETY: Surface and phys_dev are derived from same instance
            let queue_family_indexes = unsafe { QueueFamilyIndicesOpt::find(phys_dev, surface) };

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
