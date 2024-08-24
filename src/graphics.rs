use std::{
    collections::HashSet,
    ffi::CStr,
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::{offset_of, size_of, size_of_val},
    path::Path,
    rc::Rc,
    sync::Arc,
};

use ash::{
    vk::{
        api_version_major, api_version_minor, api_version_patch,
        ApplicationInfo, AttachmentDescription, AttachmentLoadOp,
        AttachmentReference, AttachmentStoreOp, BlendFactor, BufferCreateInfo,
        BufferUsageFlags, ColorComponentFlags, CommandBufferLevel,
        CommandPoolCreateFlags, CommandPoolCreateInfo, CullModeFlags,
        DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DescriptorPoolCreateInfo,
        DescriptorPoolSize, DescriptorSetAllocateInfo,
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, Format,
        FrontFace, GraphicsPipelineCreateInfo, ImageLayout, InstanceCreateInfo,
        LogicOp, MemoryPropertyFlags, PhysicalDevice, PhysicalDeviceType,
        PipelineBindPoint, PipelineColorBlendAttachmentState,
        PipelineColorBlendStateCreateInfo,
        PipelineInputAssemblyStateCreateInfo, PipelineLayoutCreateInfo,
        PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo,
        PolygonMode, PrimitiveTopology, QueueFlags, RenderPassCreateInfo,
        Result as VkResult, SampleCountFlags, ShaderStageFlags, SharingMode,
        SubpassDescription, VertexInputAttributeDescription,
        VertexInputBindingDescription, VertexInputRate, API_VERSION_1_0,
    },
    LoadingError,
};
use buffers::ManagedMappableBuffer;
use command_buffers::CommandPool;
use debug_messenger::DebugMessenger;
use descriptor_sets::{DescriptorPool, DescriptorSet, DescriptorSetLayout};
use device::Device;
use instance::Instance;

use pipeline::Pipeline;
use pipeline_layout::PipelineLayout;
use render_pass::RenderPass;
use shader_module::ShaderModule;
use structopt::StructOpt;
use strum::EnumString;

use surface::Surface;
use swapchain::Swapchain;
use vek::{Mat4, Vec2, Vec3};
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::ActiveEventLoop,
    raw_window_handle::HasDisplayHandle,
    window::{Window, WindowAttributes, WindowId},
};

const DEFAULT_WINDOW_WIDTH: u32 = 1280;

const DEFAULT_WINDOW_HEIGHT: u32 = 720;

mod buffers;
mod command_buffers;
mod debug_messenger;
mod descriptor_sets;
mod device;
mod instance;
mod pipeline;
mod pipeline_layout;
mod render_pass;
mod shader_module;
mod surface;
mod swapchain;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const VERTICES: &[Vertex] = &[
    Vertex {
        pos: Vec2::new(0.0, -0.5),
        col: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec2::new(0.5, 0.5),
        col: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec2::new(-0.5, -0.5),
        col: Vec3::new(0.0, 1.0, 0.0),
    },
];

const INDICES: &[u16] = &[0, 1, 2];

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Uniform {
    model: Mat4<f32>,
    view: Mat4<f32>,
    proj: Mat4<f32>,
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Vertex {
    pos: vek::Vec2<f32>,
    col: Vec3<f32>,
}
impl Vertex {
    fn vertex_attribute_descriptions(
        binding: u32,
    ) -> [VertexInputAttributeDescription; 2] {
        [
            VertexInputAttributeDescription::default()
                .location(0)
                .binding(binding)
                .format(Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            VertexInputAttributeDescription::default()
                .location(1)
                .binding(binding)
                .offset(offset_of!(Vertex, col) as u32)
                .format(Format::R32G32B32_SFLOAT),
        ]
    }
    fn vertex_binding_descriptions(
        binding: u32,
        input_rate: VertexInputRate,
    ) -> [VertexInputBindingDescription; 1] {
        [VertexInputBindingDescription::default()
            .binding(binding)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(input_rate)]
    }
}

//SAFETY: All members must be manually drop so we can control the Drop order in
//our Drop implementation. There are ways around this but they require more
//magic
#[derive(Debug)]
pub struct Context {
    win: Arc<Window>,
    _instance: Arc<Instance>,
    _debug_messenger: Option<DebugMessenger>,
    _surface_derived: Option<SurfaceDerived>,
    _command_buffers: Vec<command_buffers::CommandBuffer>,
    _descriptor_sets: Vec<DescriptorSet>,
    _vertex_buffers: Vec<ManagedMappableBuffer<Vertex>>,
    _uniform_buffers: Vec<ManagedMappableBuffer<Uniform>>,
    _index_buffers: Vec<ManagedMappableBuffer<u16>>,
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: ValidationLevel,
    pub dimensions: Option<Size>,
}

#[derive(Debug, StructOpt, Default, PartialEq, Eq, EnumString, Clone, Copy)]
pub enum ValidationLevel {
    #[default]
    None,
    Error,
    Warn,
    Info,
    Verbose,
}

pub type PhantomUnsendUnsync = PhantomData<*const ()>;

#[derive(Debug)]
struct SurfaceDerived {
    _swapchain: Arc<Swapchain>,
    _pipeline: Pipeline,
    _swapchain_framebuffers: Vec<swapchain::SwapchainFramebuffer>,
}
impl SurfaceDerived {
    fn new(
        device: &Arc<Device>,
        surface: Arc<Surface>,
        graphics_queue_index: u32,
        present_queue_index: u32,
        shader_modules: &[&ShaderModule],
        pipeline_layout: PipelineLayout,
    ) -> Result<Self, RenderSetupError> {
        use RenderSetupError::*;
        let swapchain = Arc::new(
            //SAFETY: Device and surface are from the same instance
            unsafe {
                Swapchain::new(
                    device,
                    &surface,
                    present_queue_index,
                    graphics_queue_index,
                    None,
                )
            }
            .map_err(|_| SwapchainCreation)?,
        );
        let viewports = [swapchain.default_viewport()];
        let scissors = [swapchain.default_scissor()];
        let color_attachment = AttachmentDescription::default()
            .format(swapchain.get_format())
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref: AttachmentReference =
            AttachmentReference::default()
                .attachment(0)
                .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let subpass = SubpassDescription::default()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments);
        let subpasses = &[subpass];
        let attachments = [color_attachment];

        let _viewport_state = PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let render_pass_ci = RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(subpasses);

        //SAFETY: Valid ci
        let render_pass = unsafe { RenderPass::new(device, &render_pass_ci) }
            .map_err(|e| {
            UnknownVulkan("creating render pass".to_owned(), e)
        })?;

        let shader_stages = shader_modules
            .iter()
            .map(|m| {
                PipelineShaderStageCreateInfo::default()
                    .module(m.as_raw())
                    .stage(m.get_stage())
                    .name(m.get_name())
            })
            .collect::<Vec<_>>();
        let vertex_attribute_descriptions =
            Vertex::vertex_attribute_descriptions(0);
        let vertex_binding_descriptions =
            Vertex::vertex_binding_descriptions(0, VertexInputRate::VERTEX);
        let _vertex_input_state = PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_binding_descriptions);
        let _input_assembly_state =
            PipelineInputAssemblyStateCreateInfo::default()
                .topology(PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
        let _input_assembly_state =
            PipelineInputAssemblyStateCreateInfo::default()
                .topology(PrimitiveTopology::TRIANGLE_STRIP)
                .primitive_restart_enable(false);

        let _rasterization_state =
            PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(CullModeFlags::BACK)
                .front_face(FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false);
        let _multisample_state = PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(SampleCountFlags::TYPE_1);

        let attachments = [PipelineColorBlendAttachmentState::default()
            .dst_color_blend_factor(BlendFactor::ONE)
            .color_write_mask(ColorComponentFlags::RGBA)];
        let _color_blend_state = PipelineColorBlendStateCreateInfo::default()
            .attachments(&attachments)
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let pipeline_ci = GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&_vertex_input_state)
            .input_assembly_state(&_input_assembly_state)
            .viewport_state(&_viewport_state)
            .rasterization_state(&_rasterization_state)
            .multisample_state(&_multisample_state)
            .color_blend_state(&_color_blend_state)
            .layout(pipeline_layout.as_inner())
            .render_pass(render_pass.as_inner())
            .subpass(0);
        let pipeline =
            //SAFETY: valid cis
            unsafe { Pipeline::new_graphics_pipelines(device, &[pipeline_ci]) }
                .map_err(|e| {
                    UnknownVulkan("creating graphics pipeline".into(), e)
                })?
                .pop()
                .expect("How did this not error yet return 0 pipelines?");
        let swapchain_framebuffers = swapchain
            .create_compatible_framebuffers(&render_pass)
            .map_err(|e| {
                UnknownVulkan("While creating framebuffers".to_owned(), e)
            })?;
        Ok(Self {
            _swapchain: swapchain,
            _pipeline: pipeline,
            _swapchain_framebuffers: swapchain_framebuffers,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RenderSetupError {
    #[error("Could not create swapchain")]
    SwapchainCreation,
    #[error("Unknown vulkan error while {0}. Error code {1:?}")]
    UnknownVulkan(String, VkResult),
}

#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum ContextCreationError {
    #[error("Could not load Vulkan {0:?}")]
    Loading(LoadingError),
    #[error("Instance creation failed")]
    InstanceCreation,
    #[error("Missing mandatory extensions {0:?}")]
    MissingMandatoryExtensions(Vec<String>),
    #[error("No suitable devices")]
    NoSuitableDevice,
    #[error("Could not create device")]
    DeviceCreation,
    #[error("Could not create surface")]
    SurfaceCreation,

    #[error("Could not create command buffers")]
    CommandBufferCreation,
    #[error("Unknown vulkan error while {0}. Error code {1:?}")]
    UnknownVulkan(String, VkResult),
    #[error("Error creating window: {0:?}")]
    WindowCreation(#[from] winit::error::OsError),
    #[error("Could not make shader compiler")]
    ShaderCompilerCreation,
    #[error("Compilation errors\n{0}")]
    ShaderLoading(#[from] ShaderModuleCreationErrors),
    #[error("Render setup error: {0}")]
    RenderSetup(#[from] RenderSetupError),
    #[error("Error making descriptor set")]
    DescriptorSetCreation,
}

#[derive(thiserror::Error, Debug)]
pub struct ShaderModuleCreationErrors(Vec<shader_module::Error>);

impl Display for ShaderModuleCreationErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for error in &self.0 {
            let line = match error {
                shader_module::Error::FileLoad(file_name, err) => {
                    format!(
                        "{} not found: Error code: {}",
                        file_name.to_string_lossy(),
                        err
                    )
                }
                shader_module::Error::ShaderCompilation(err) => err.to_string(),
                shader_module::Error::MemoryExhaustion => {
                    "Memory exhausted".into()
                }
            };
            f.write_str(&line)?;
        }
        Ok(())
    }
}

const KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(
        event_loop: &ActiveEventLoop,
        mut opts: ContextCreateOpts,
    ) -> Result<Self, ContextCreationError> {
        use ContextCreationError::*;
        let entry =
        //SAFETY: Must be dropped *after* instance, which we accomplish by
        //having instance hold on to a ref counted pointer to entry
            Arc::new(unsafe { ash::Entry::load().map_err(Loading) }?);
        //SAFETY: Should be good?
        let vk_version = match unsafe { entry.try_enumerate_instance_version() }
        {
            Ok(ver) => ver.unwrap_or(API_VERSION_1_0),

            Err(_) => unreachable!(),
        };
        log::info!(
            "Available vk version {}.{}.{}",
            api_version_major(vk_version),
            api_version_minor(vk_version),
            api_version_patch(vk_version)
        );
        let app_info = ApplicationInfo::default()
            .api_version(vk_version)
            .application_name(c"arpgn");
        let avail_extensions =
        //SAFETY: Should always be good
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| {
                    UnknownVulkan("enumerating instance extensions".to_owned(), e)
                })?;
        let windowing_required_extensions =
            ash_window::enumerate_required_extensions(
                event_loop.display_handle().unwrap().as_raw(),
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
            return Err(MissingMandatoryExtensions(
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

        let mut instance_create_info = InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_exts)
            .enabled_layer_names(&layer_names);

        let all_debug_message_types =
            DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                | DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION;

        let mut debug_utils_messenger_ci =
            DebugUtilsMessengerCreateInfoEXT::default()
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
        let instance = Arc::new(
            //SAFETY: Valid ci. We know because we made it and none of the lifetimes
            //involved have expired
            unsafe { Instance::new(&entry, &instance_create_info) }
                .map_err(|_| InstanceCreation)?,
        );

        let _debug_messenger = if opts.graphics_validation_layers
            != ValidationLevel::None
        {
            //SAFETY: Valid ci. We know cause we made it
            unsafe { DebugMessenger::new(&instance, &debug_utils_messenger_ci) }
                .ok()
        } else {
            None
        };

        let win = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default().with_inner_size(
                        opts.dimensions.unwrap_or(
                            LogicalSize::new(
                                DEFAULT_WINDOW_WIDTH,
                                DEFAULT_WINDOW_HEIGHT,
                            )
                            .into(),
                        ),
                    ),
                )
                .map_err(WindowCreation)?,
        );

        let surface = surface::Surface::new(&instance, &win)
            .map_err(|_| SurfaceCreation)?;

        let physical_devices = instance
            .get_physical_devices()
            .expect("Should always succeed");

        let scored_phys_dev = physical_devices
            .iter()
            .fold(None, |best_found, current_dev| {
                //SAFETY: We derived current_dev from device
                let score = unsafe {
                    evaluate_physical_device(&instance, *current_dev, &surface)
                };
                if score > best_found {
                    score
                } else {
                    best_found
                }
            })
            .ok_or(NoSuitableDevice)?;
        //TODO: Properly check for these extensions
        let device_extension_names = vec![ash::khr::swapchain::NAME.as_ptr()];
        let mut queue_family_indices = HashSet::with_capacity(2);
        queue_family_indices.insert(scored_phys_dev.present_queue_index);
        queue_family_indices.insert(scored_phys_dev.graphics_queue_index);
        let queue_create_infos: Vec<_> = queue_family_indices
            .iter()
            .map(|qfi| {
                DeviceQueueCreateInfo::default()
                    .queue_family_index(*qfi)
                    .queue_priorities(&[1.0])
            })
            .collect();
        #[allow(deprecated)]
        let dev_ci = DeviceCreateInfo::default()
            .enabled_extension_names(&device_extension_names)
            .queue_create_infos(&queue_create_infos)
            //We're using this to be compatible with older implementations. This
            //isn't necessary but it's like. 2 values in the struct that would
            //have to be zeroed anyways. Who cares.
            .enabled_layer_names(&layer_names);
        let device = Arc::new(
            //SAFETY: valid ci and phys_dev is derived from instance. We
            //accomplished these
            unsafe {
                Device::new(&instance, scored_phys_dev.phys_dev, &dev_ci)
            }
            .map_err(|_| DeviceCreation)?,
        );

        let shader_compiler =
            shaderc::Compiler::new().ok_or(ShaderCompilerCreation)?;

        let vert_shader_path = Path::new("shaders/shader.vert");

        let vert_shader_mod = ShaderModule::new(
            &device,
            &shader_compiler,
            vert_shader_path,
            ShaderStageFlags::VERTEX,
            "main",
            None,
        );
        let frag_shader_path = Path::new("shaders/shader.frag");
        let frag_shader_mod = ShaderModule::new(
            &device,
            &shader_compiler,
            frag_shader_path,
            ShaderStageFlags::FRAGMENT,
            "main",
            None,
        );

        let (vert_shader_mod, frag_shader_mod) =
            match (vert_shader_mod, frag_shader_mod) {
                (Ok(mod1), Ok(mod2)) => Ok((mod1, mod2)),
                (Err(e), Ok(_)) | (Ok(_), Err(e)) => {
                    Err(ShaderModuleCreationErrors(vec![e]))
                }
                (Err(e1), Err(e2)) => {
                    Err(ShaderModuleCreationErrors(vec![e1, e2]))
                }
            }?;
        let bindings = &[DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)];
        let descriptor_set_layout_ci =
            DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        //SAFETY: Valid ci
        let descriptor_set_layout = unsafe {
            DescriptorSetLayout::new(&device, &descriptor_set_layout_ci)
        }
        .map_err(|_| ContextCreationError::DescriptorSetCreation)?;

        let descriptor_set_layouts = &[descriptor_set_layout.as_inner()];

        let pool_sizes = &[DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT)
            .ty(DescriptorType::UNIFORM_BUFFER)];
        let descriptor_pool_ci = DescriptorPoolCreateInfo::default()
            .max_sets(MAX_FRAMES_IN_FLIGHT)
            .pool_sizes(pool_sizes);
        let descriptor_pool =
            //SAFETY: valid ci
            Rc::new(unsafe { DescriptorPool::new(&device, &descriptor_pool_ci) }
                            .map_err(|_| DescriptorSetCreation)?);

        let duped_layouts = vec![
            descriptor_set_layout.as_inner();
            MAX_FRAMES_IN_FLIGHT as usize
        ];
        let descriptor_set_ai = DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.as_inner())
            .set_layouts(&duped_layouts);

        //SAFETY: valid ai
        let descriptor_sets = unsafe {
            DescriptorSet::alloc(&descriptor_pool, &descriptor_set_ai)
        }
        .unwrap();
        let pipeline_layout_ci = PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts);

        let pipeline_layout =
            //SAFETY: Valid ci
            unsafe { PipelineLayout::new(&device, &pipeline_layout_ci) }
                .map_err(|e| {
                    UnknownVulkan(
                        "creating pipeline layout".into(),
                        e,
                    )
                })?;
        let command_pool_ci = CommandPoolCreateInfo::default()
            .queue_family_index(scored_phys_dev.graphics_queue_index)
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = Rc::new(
            //SAFETY: Valid ci
            unsafe { CommandPool::new(&device, &command_pool_ci) }
                .map_err(|_| CommandBufferCreation)?,
        );

        let command_buffers = command_pool
            .alloc_command_buffers(
                MAX_FRAMES_IN_FLIGHT,
                CommandBufferLevel::PRIMARY,
            )
            .map_err(|_| CommandBufferCreation)?;

        let mut vertex_buffers =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());
        let mut index_buffers =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());
        let mut uniform_buffers =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());
        let buffer_qfis = &[scored_phys_dev.graphics_queue_index];

        let vertex_buffer_ci = BufferCreateInfo::default()
            .sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(buffer_qfis)
            .usage(BufferUsageFlags::VERTEX_BUFFER)
            .size(size_of_val(VERTICES) as u64);

        let vertex_buffer_ai = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            required_flags: MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let index_buffer_ci = BufferCreateInfo::default()
            .sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(buffer_qfis)
            .usage(BufferUsageFlags::INDEX_BUFFER)
            .size(size_of_val(INDICES) as u64);

        let index_buffer_ai = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            required_flags: MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let uniform_buffer_ci = BufferCreateInfo::default()
            .sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(buffer_qfis)
            .usage(BufferUsageFlags::UNIFORM_BUFFER)
            .size(size_of::<Uniform>() as u64);

        let uniform_buffer_ai = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            required_flags: MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            vertex_buffers.push(
                //SAFETY: valid ci and ai
                unsafe {
                    ManagedMappableBuffer::new(
                        &device,
                        &vertex_buffer_ci,
                        &vertex_buffer_ai,
                    )
                }
                .unwrap(),
            );
            index_buffers.push(
                //SAFETY: valid ci and ai
                unsafe {
                    ManagedMappableBuffer::new(
                        &device,
                        &index_buffer_ci,
                        &index_buffer_ai,
                    )
                }
                .unwrap(),
            );
            uniform_buffers.push(
                //SAFETY: valid ci and ai
                unsafe {
                    ManagedMappableBuffer::new(
                        &device,
                        &uniform_buffer_ci,
                        &uniform_buffer_ai,
                    )
                }
                .unwrap(),
            );
        }
        Ok(Context {
            win,
            _command_buffers: command_buffers,
            _descriptor_sets: descriptor_sets,
            _instance: instance,
            _debug_messenger,
            _surface_derived: Some(SurfaceDerived::new(
                &device,
                Arc::new(surface),
                scored_phys_dev.graphics_queue_index,
                scored_phys_dev.present_queue_index,
                &[&vert_shader_mod, &frag_shader_mod],
                pipeline_layout,
            )?),
            _vertex_buffers: vertex_buffers,
            _uniform_buffers: uniform_buffers,
            _index_buffers: index_buffers,
        })
    }
    pub fn resize(&mut self) {}
    pub fn draw(&mut self) {}

    pub fn win_id(&self) -> WindowId {
        self.win.id()
    }
}

struct ScoredPhysDev {
    score: u64,
    phys_dev: PhysicalDevice,
    graphics_queue_index: u32,
    present_queue_index: u32,
}
impl PartialEq for ScoredPhysDev {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl PartialOrd for ScoredPhysDev {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> std::option::Option<std::cmp::Ordering> {
        Some(self.score.cmp(&other.score))
    }
}

impl Ord for ScoredPhysDev {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for ScoredPhysDev {}

//SAFETY REQUIREMENTS: phys_dev must be derived from instance, surface must be
//derived from instance.
unsafe fn evaluate_physical_device(
    instance: &Instance,
    phys_dev: PhysicalDevice,
    surface: &Surface,
) -> Option<ScoredPhysDev> {
    //SAFETY:phys_dev must be derived from instance
    let properties =
        unsafe { instance.get_relevant_physical_device_properties(phys_dev) };
    let graphics_queue_index = properties
        .queue_families
        .iter()
        .enumerate()
        .find(|(qfi, qfp)| {
            qfp.queue_flags.intersects(QueueFlags::GRAPHICS)
            //SAFETY: qfi is in bounds, phys_dev comes from same instance as
            //surface
                && unsafe {
                    surface
                        .does_queue_support_presentation(phys_dev, *qfi as u32)
                }
        })
        .or_else(|| {
            properties
                .queue_families
                .iter()
                .enumerate()
                .find(|(_, qfp)| {
                    qfp.queue_flags.intersects(QueueFlags::GRAPHICS)
                })
        })
        .map(|(i, _)| i as u32);
    let present_queue_index = graphics_queue_index
        .filter(|graphics_queue_index| {
            //SAFETY: qfi is in bounds, phys_dev comes from same instance as surface
            unsafe {
                surface.does_queue_support_presentation(
                    phys_dev,
                    *graphics_queue_index,
                )
            }
        })
        .or_else(|| {
            properties
                .queue_families
                .iter()
                .copied()
                .enumerate()
                .find(|(qfi, _props)| {
                    //SAFETY: qfi is in bounds, phys_dev comes from same
                    //instance as surface
                    unsafe {
                        surface.does_queue_support_presentation(
                            phys_dev,
                            *qfi as u32,
                        )
                    }
                })
                .map(|(i, _)| i as u32)
        });
    if properties.extensions.iter().any(|ext| {
        ash::khr::swapchain::NAME.eq(ext.extension_name_as_c_str().expect(
            "It'd be weird if we got an extension that wasn't a valid cstr",
        ))
        //SAFETY: phys_dev from same source as surface
    }) && unsafe {
        surface.get_compatible_swapchain_info(phys_dev).ok().map_or(
            false,
            |swap_info| {
                !swap_info.formats.is_empty()
                    && !swap_info.present_modes.is_empty()
            },
        )
    } {
        graphics_queue_index.and_then(|graphics_queue_index| {
            present_queue_index.map(|present_queue_index| {
                //Weight CPUs more highly based on what type they are. If unknown,
                //just return some weird low value but better than CPU
                let device_type_score = match properties.props.device_type {
                    PhysicalDeviceType::DISCRETE_GPU => 1000,
                    PhysicalDeviceType::INTEGRATED_GPU => 500,
                    PhysicalDeviceType::CPU => 0,
                    _ => 100,
                };
                //If multiple of same category, pick the one with a shared graphics and presentation queue
                let queue_score = if graphics_queue_index == present_queue_index
                {
                    100
                } else {
                    50
                };
                ScoredPhysDev {
                    score: queue_score + device_type_score,
                    phys_dev,
                    graphics_queue_index,
                    present_queue_index,
                }
            })
        })
    } else {
        None
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
        ValidationLevel::Warn => warning,
        ValidationLevel::Info => info,
        ValidationLevel::Verbose => verbose,
    }
}
