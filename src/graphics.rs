use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::{offset_of, size_of, size_of_val},
    ops::MulAssign,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

use ash::{
    prelude::VkResult,
    vk::{
        self, api_version_major, api_version_minor, api_version_patch, AccessFlags,
        ApplicationInfo, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentStoreOp, BlendFactor, BlendOp, BorderColor, BufferImageCopy, BufferUsageFlags,
        ClearColorValue, ClearDepthStencilValue, ClearValue, ColorComponentFlags,
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags,
        CommandPoolCreateFlags, CommandPoolCreateInfo, CompareOp, CullModeFlags,
        DebugUtilsLabelEXT, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DependencyFlags, DescriptorBufferInfo,
        DescriptorImageInfo, DescriptorPoolCreateInfo, DescriptorPoolSize,
        DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, Extent3D, Filter, Format,
        FormatFeatureFlags, FrontFace, GraphicsPipelineCreateInfo, ImageAspectFlags, ImageBlit,
        ImageCreateInfo, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers,
        ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView,
        ImageViewCreateInfo, ImageViewType, IndexType, InstanceCreateInfo, LogicOp, Offset3D,
        PhysicalDevice, PhysicalDeviceType, PipelineBindPoint, PipelineColorBlendAttachmentState,
        PipelineColorBlendStateCreateInfo, PipelineDepthStencilStateCreateInfo,
        PipelineInputAssemblyStateCreateInfo, PipelineLayoutCreateInfo,
        PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo,
        PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineVertexInputStateCreateInfo,
        PipelineViewportStateCreateInfo, PolygonMode, PrimitiveTopology, PushConstantRange,
        QueueFlags, RenderPassBeginInfo, RenderPassCreateInfo, SampleCountFlags, Sampler,
        SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, SharingMode,
        SubpassContents, SubpassDescription, VertexInputAttributeDescription,
        VertexInputBindingDescription, VertexInputRate, WriteDescriptorSet, API_VERSION_1_0,
        QUEUE_FAMILY_IGNORED,
    },
    LoadingError,
};
use buffers::MappableBuffer;
use clap::Parser;
use command_buffers::{CommandBuffer, CommandPool};
use descriptor_sets::{DescriptorPool, DescriptorSet, DescriptorSetLayout};
use device::Device;
use instance::Instance;

use crate::utils::log2_floor;

use super::utils::debug_string;
use pipeline::Pipeline;
use pipeline_layout::PipelineLayout;
use render_pass::RenderPass;
use shader_module::ShaderModule;
use strum::EnumString;
use utils::FenceProducer;

use surface::Surface;
use swapchain::Swapchain;
use sync_objects::{Fence, Semaphore};
use utils::associate_debug_name;
use vek::{
    num_traits::{One, Zero},
    Mat4, Vec2, Vec3, Vec4,
};
use vk_mem::{Alloc, MemoryUsage};
use winit::{
    dpi::{LogicalSize, PhysicalSize, Size},
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
mod sync_objects;
mod utils;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const VERTICES: &[Vertex] = &[
    Vertex {
        pos: Vec3::new(-0.5, -0.5, 0.0),
        col: Vec3::new(0.0, 0.0, 1.0),
        coord: Vec2::new(0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(-0.5, 0.5, 0.0),
        col: Vec3::new(0.0, 1.0, 1.0),
        coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.5, 0.5, 0.0),
        col: Vec3::new(1.0, 1.0, 1.0),
        coord: Vec2::new(1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.5, -0.5, 0.0),
        col: Vec3::new(1.0, 0.0, 1.0),
        coord: Vec2::new(1.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(-0.5, -0.5, -1.0),
        col: Vec3::new(0.0, 0.0, 1.0),
        coord: Vec2::new(0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(-0.5, 0.5, -1.0),
        col: Vec3::new(0.0, 1.0, 1.0),
        coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.5, 0.5, -1.0),
        col: Vec3::new(1.0, 1.0, 1.0),
        coord: Vec2::new(1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.5, -0.5, -1.0),
        col: Vec3::new(1.0, 0.0, 1.0),
        coord: Vec2::new(1.0, 1.0),
    },
];

const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct UniformBuffer {
    view: Mat4<f32>,
    proj: Mat4<f32>,
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Vertex {
    pos: vek::Vec3<f32>,
    col: Vec3<f32>,
    coord: Vec2<f32>,
}
impl Vertex {
    fn vertex_attribute_descriptions(binding: u32) -> [VertexInputAttributeDescription; 3] {
        [
            VertexInputAttributeDescription::default()
                .location(0)
                .binding(binding)
                .format(Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            VertexInputAttributeDescription::default()
                .location(1)
                .binding(binding)
                .offset(offset_of!(Vertex, col) as u32)
                .format(Format::R32G32B32_SFLOAT),
            VertexInputAttributeDescription::default()
                .location(2)
                .binding(binding)
                .offset(offset_of!(Vertex, coord) as u32)
                .format(Format::R32G32_SFLOAT),
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

impl Drop for Context {
    fn drop(&mut self) {
        //SAFETY: Almost always fine
        unsafe { self.device.as_inner_ref().device_wait_idle() }.unwrap();
    }
}

//SAFETY: All members must be manually drop so we can control the Drop order in
//our Drop implementation. There are ways around this but they require more
//magic
#[derive(Debug)]
pub struct Context {
    win: Arc<Window>,
    _instance: Arc<Instance>,

    surface_derived: Option<SurfaceDerived>,
    command_buffers: Vec<command_buffers::CommandBuffer>,
    uniform_descriptor_sets: Vec<DescriptorSet>,
    vertex_buffers: Vec<MappableBuffer<Vertex>>,
    //TODO: Apparently this uniform buffer is captured by the descriptor set
    uniform_buffers: Vec<MappableBuffer<UniformBuffer>>,
    index_buffers: Vec<MappableBuffer<u16>>,
    image_available_semaphores: Vec<Semaphore>,
    prev_frame_finished_fences: Vec<Fence>,
    frame_index: usize,
    render_complete_semaphores: Vec<Semaphore>,
    graphics_queue_index: u32,
    present_queue_index: u32,
    pipeline_layout: PipelineLayout,
    device: Arc<Device>,
    vert_shader_mod: ShaderModule,
    frag_shader_mod: ShaderModule,
    start_time: Instant,
    _gpu_image_view: GpuImageView,
    _texture_sampler: TextureSampler,
    multisample_flag: SampleCountFlags,
    minimized: bool,
}

#[derive(Debug, Default)]
pub struct ContextCreateOpts {
    pub graphics_validation_layers: ValidationLevel,
    pub dimensions: Option<Size>,
    pub unified_transfer_graphics_queue: bool,
    pub debuggable_shaders: bool,
}

#[derive(Debug, Parser, Default, PartialEq, Eq, EnumString, Clone, Copy)]
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
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,

    pipeline: Pipeline,
    swapchain_framebuffers: Vec<swapchain::SwapchainFramebuffer>,
    render_pass: RenderPass,
}
impl SurfaceDerived {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &Arc<Device>,
        surface: Arc<Surface>,
        graphics_queue_index: u32,
        present_queue_index: u32,
        shader_modules: &[&ShaderModule],
        pipeline_layout: &PipelineLayout,
        multisample_flag: SampleCountFlags,
        old_swapchain: Option<&Arc<Swapchain>>,
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
                    old_swapchain,
                    debug_string!(device.is_debug(), "Main Swapchain"),
                )
            }
            .map_err(|_| SwapchainCreation)?,
        );
        log::trace!("Swapchain Format: {:?}", swapchain.get_format());
        log::trace!(
            "Swapchain Dimensions: {} {} {}",
            swapchain.default_viewport().width,
            swapchain.default_viewport().height,
            swapchain.get_aspect_ratio()
        );
        let candidate_depth_formats = [Format::D32_SFLOAT, Format::D32_SFLOAT_S8_UINT];

        let depth_format = candidate_depth_formats
            .iter()
            .copied()
            .find(|f| {
                let props = device.get_format_properties(*f);
                props
                    .optimal_tiling_features
                    .contains(FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .ok_or(RenderSetupError::NoSupportedDepthFormat)?;
        log::trace!("Depth Format: {:?}", depth_format);
        let viewports = [swapchain.default_viewport()];
        let scissors = [swapchain.as_rect()];
        let color_attachment = AttachmentDescription::default()
            .format(swapchain.get_format())
            .samples(multisample_flag)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_resolve_attachment = vk::AttachmentDescription::default()
            .format(swapchain.get_format())
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(multisample_flag)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_ref = AttachmentReference::default()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_resolve_attachment_ref = AttachmentReference::default()
            .attachment(2)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachment_ref = AttachmentReference::default()
            .attachment(1)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let resolve_attachments = &[color_resolve_attachment_ref];

        let subpass = SubpassDescription::default()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .resolve_attachments(resolve_attachments)
            .depth_stencil_attachment(&depth_attachment_ref);
        let subpasses = &[subpass];
        let attachments = [color_attachment, depth_attachment, color_resolve_attachment];

        let msaa_color_attachment_ci = ImageCreateInfo::default()
            .extent(
                Extent3D::default()
                    .depth(1)
                    .width(swapchain.width())
                    .height(swapchain.height()),
            )
            .samples(multisample_flag)
            .format(swapchain.get_format())
            .tiling(ImageTiling::OPTIMAL)
            .usage(ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .mip_levels(1)
            .array_layers(1)
            .image_type(ImageType::TYPE_2D);

        let msaa_color_attachment_ai = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,

            ..Default::default()
        };

        let depth_attachment_ci = msaa_color_attachment_ci
            .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .format(depth_format);
        let depth_attachment_ai = msaa_color_attachment_ai.clone();

        let mut msaa_color_attachment_views = Vec::with_capacity(swapchain.image_count() as usize);
        let mut depth_attachment_views = Vec::with_capacity(swapchain.image_count() as usize);

        for i in 0..swapchain.image_count() {
            let msaa_color_attachment_image = Arc::new(
                unsafe {
                    GpuImage::new(
                        device,
                        &msaa_color_attachment_ci,
                        &msaa_color_attachment_ai,
                        debug_string!(
                            device.is_debug(),
                            "Multisample Color Attachment Image [{}]",
                            i
                        ),
                    )
                }
                .map_err(|e| {
                    RenderSetupError::UnknownVulkan(
                        format!("Could not create Multisample Color Attachment Image {}", i),
                        e,
                    )
                })?,
            );
            let msaa_color_attachment_subresource_range = ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let msaa_color_attachment_view_ci = ImageViewCreateInfo::default()
                .image(msaa_color_attachment_image.inner)
                .format(swapchain.get_format())
                .subresource_range(msaa_color_attachment_subresource_range)
                .view_type(ImageViewType::TYPE_2D);
            let msaa_color_attachment_image_view = unsafe {
                GpuImageView::new(
                    &msaa_color_attachment_image,
                    &msaa_color_attachment_view_ci,
                    debug_string!(
                        device.is_debug(),
                        "Multisample Attachment Image View [{}]",
                        i
                    ),
                )
            }
            .map_err(|e| {
                RenderSetupError::UnknownVulkan(
                    format!("Could not create MSAA Color Attachment Image View [{}]", i),
                    e,
                )
            })?;
            msaa_color_attachment_views.push(msaa_color_attachment_image_view);
            let depth_attachment_image = Arc::new(
                unsafe {
                    GpuImage::new(
                        device,
                        &depth_attachment_ci,
                        &depth_attachment_ai,
                        debug_string!(device.is_debug(), "Depth Attachment Image [{}]", i),
                    )
                }
                .map_err(|e| {
                    RenderSetupError::UnknownVulkan(
                        format!("Could not create Multisample Color Attachment Image {}", i),
                        e,
                    )
                })?,
            );
            let depth_attachment_subresource_range = ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let depth_attachment_view_ci = ImageViewCreateInfo::default()
                .image(depth_attachment_image.inner)
                .format(depth_format)
                .subresource_range(depth_attachment_subresource_range)
                .view_type(ImageViewType::TYPE_2D);
            let depth_attachment_image_view = unsafe {
                GpuImageView::new(
                    &depth_attachment_image,
                    &depth_attachment_view_ci,
                    debug_string!(device.is_debug(), "Depth Attachment Image View [{}]", i),
                )
            }
            .map_err(|e| {
                RenderSetupError::UnknownVulkan(
                    format!("Could not create MSAA Color Attachment Image View [{}]", i),
                    e,
                )
            })?;
            depth_attachment_views.push(depth_attachment_image_view);
        }

        let viewport_state = PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let render_pass_ci = RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(subpasses);

        //SAFETY: Valid ci
        let render_pass = unsafe {
            RenderPass::new(
                device,
                &render_pass_ci,
                debug_string!(device.is_debug(), "Main render pass"),
            )
        }
        .map_err(|e| UnknownVulkan("creating render pass".to_owned(), e))?;

        let shader_stages = shader_modules
            .iter()
            .map(|m| {
                PipelineShaderStageCreateInfo::default()
                    .module(m.as_raw())
                    .stage(m.get_stage())
                    .name(m.get_name())
            })
            .collect::<Vec<_>>();
        let vertex_attribute_descriptions = Vertex::vertex_attribute_descriptions(0);
        let vertex_binding_descriptions =
            Vertex::vertex_binding_descriptions(0, VertexInputRate::VERTEX);
        let vertex_input_state = PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_binding_descriptions);

        let input_assembly_state = PipelineInputAssemblyStateCreateInfo::default()
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let rasterization_state = PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false);
        let multisample_state =
            PipelineMultisampleStateCreateInfo::default().rasterization_samples(multisample_flag);

        let attachments = [PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(ColorComponentFlags::RGBA)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .alpha_blend_op(BlendOp::ADD)];
        let color_blend_state = PipelineColorBlendStateCreateInfo::default()
            .attachments(&attachments)
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);
        let depth_stencil_state = PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(CompareOp::LESS)
            .stencil_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let pipeline_ci = GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(pipeline_layout.get_inner())
            .render_pass(render_pass.get_inner())
            .subpass(0);
        let pipeline = unsafe {
            Pipeline::new_graphics_pipelines(
                device,
                &[pipeline_ci],
                Some(|_, _| Some("Graphics Pipeline".into())),
            )
        }
        .map_err(|e| UnknownVulkan("creating graphics pipeline".into(), e))?
        .pop()
        .expect("How did this not error yet return 0 pipelines?");
        let swapchain_framebuffers = swapchain
            .create_compatible_framebuffers(
                &render_pass,
                Some(msaa_color_attachment_views),
                Some(depth_attachment_views),
                Some(|i, _| Some(format!("Swapchain framebuffer {}", i))),
            )
            .map_err(|e| UnknownVulkan("While creating framebuffers".to_owned(), e))?;
        Ok(Self {
            swapchain,
            pipeline,
            swapchain_framebuffers,
            render_pass,
            surface,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RenderSetupError {
    #[error("Could not create swapchain")]
    SwapchainCreation,
    #[error("Unknown vulkan error while {0}. Error code {1:?}")]
    UnknownVulkan(String, ash::vk::Result),
    #[error("No supported depth format")]
    NoSupportedDepthFormat,
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
    UnknownVulkan(String, ash::vk::Result),
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
    #[error("Couldn't find texture")]
    MissingTexture,
    #[error("Couldn't decode texture")]
    InvalidTexture,
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
                shader_module::Error::MemoryExhaustion => "Memory exhausted".into(),
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
        let vk_version = match unsafe { entry.try_enumerate_instance_version() } {
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
            .api_version(ash::vk::API_VERSION_1_0)
            .application_name(c"arpgn");
        let avail_device_extensions =
        //SAFETY: Should always be good
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| {
                    UnknownVulkan("enumerating instance extensions".to_owned(), e)
                })?;
        let windowing_required_extensions = ash_window::enumerate_required_extensions(
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

        let mut missing_mandatory_extensions = Vec::with_capacity(mandatory_extensions.len());

        for ext in &mandatory_extensions {
            let mut found = false;
            for instance_ext in &avail_device_extensions {
                if (*ext).eq(instance_ext.extension_name_as_c_str().unwrap()) {
                    found = true;
                    break;
                }
            }
            if !found {
                missing_mandatory_extensions.push(ext.to_str().unwrap().to_owned());
            }
        }

        if !missing_mandatory_extensions.is_empty() {
            return Err(MissingMandatoryExtensions(missing_mandatory_extensions));
        }
        let mut device_extension_names: Vec<_> = mandatory_extensions
            .iter()
            .map(|str| str.as_ptr())
            .collect();
        if opts.graphics_validation_layers != ValidationLevel::None
            && avail_device_extensions
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
            device_extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        } else {
            opts.graphics_validation_layers = ValidationLevel::None;
        }
        let mut layer_names = Vec::with_capacity(1);
        if opts.graphics_validation_layers != ValidationLevel::None {
            layer_names.push(KHRONOS_VALIDATION_LAYER_NAME.as_ptr());
        }

        let mut instance_create_info = InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&device_extension_names)
            .enabled_layer_names(&layer_names);

        let all_debug_message_types = DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
            | DebugUtilsMessageTypeFlagsEXT::GENERAL
            | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | DebugUtilsMessageTypeFlagsEXT::VALIDATION;

        let mut debug_utils_messenger_ci = DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(graphics_validation_sev_to_debug_utils_flags(
                opts.graphics_validation_layers,
            ))
            .message_type(all_debug_message_types)
            .pfn_user_callback(Some(debug_messenger::default_debug_callback));
        if !layer_names.is_empty() {
            instance_create_info = instance_create_info.push_next(&mut debug_utils_messenger_ci);
        }
        //SAFETY: Valid ci. We know because we made it and none of the lifetimes
        //involved have expired
        let mut instance = unsafe { Instance::new(&entry, &instance_create_info) }
            .map_err(|_| InstanceCreation)?;

        if opts.graphics_validation_layers != ValidationLevel::None {
            //SAFETY: Valid ci. We know cause we made it
            unsafe {
                instance.init_debug_messenger(&debug_utils_messenger_ci);
            }
        }

        let instance = Arc::new(instance);

        let win = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_inner_size(
                    opts.dimensions.unwrap_or(
                        LogicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT).into(),
                    ),
                ))
                .map_err(WindowCreation)?,
        );

        let surface = surface::Surface::new(&instance, &win).map_err(|_| SurfaceCreation)?;

        let physical_devices = instance
            .get_physical_devices()
            .expect("Should always succeed");

        let scored_phys_dev = physical_devices
            .iter()
            .fold(None, |best_found, current_dev| {
                //SAFETY: We derived current_dev from device
                let score = unsafe {
                    evaluate_physical_device(
                        &instance,
                        *current_dev,
                        &surface,
                        opts.unified_transfer_graphics_queue,
                    )
                };
                if score > best_found {
                    score
                } else {
                    best_found
                }
            })
            .ok_or(NoSuitableDevice)?;
        let graphics_queue_family_index = scored_phys_dev.graphics_queue_index;
        let present_queue_family_index = scored_phys_dev.present_queue_index;
        let transfer_queue_family_index = scored_phys_dev.transfer_queue_index;
        let multisample_flag = scored_phys_dev.multisample_flag;
        let mut device_extension_names = vec![ash::khr::swapchain::NAME.as_ptr()];
        log::trace!("Graphics Queue Index: {}", graphics_queue_family_index);
        log::trace!("Transfer Queue Index: {}", transfer_queue_family_index);
        log::trace!("Present Queue Index: {}", present_queue_family_index);
        //TODO: Properly check for these extensions
        let physical_device = scored_phys_dev.phys_dev;
        let avail_device_extensions = unsafe {
            instance
                .as_inner_ref()
                .enumerate_device_extension_properties(physical_device)
        }
        .unwrap();

        for mandatory_ext in device_extension_names.iter() {
            if !avail_device_extensions.iter().any(|dev_ext| {
                unsafe { CStr::from_ptr(*mandatory_ext) }
                    .eq(dev_ext.extension_name_as_c_str().unwrap())
            }) {
                missing_mandatory_extensions
                    .push(unsafe { CStr::from_ptr(*mandatory_ext).to_string_lossy().into() });
            }
        }

        if !missing_mandatory_extensions.is_empty() {
            return Err(ContextCreationError::MissingMandatoryExtensions(
                missing_mandatory_extensions,
            ));
        }

        if opts.debuggable_shaders
            && avail_device_extensions.iter().any(|device_ext| {
                ash::khr::shader_non_semantic_info::NAME
                    .eq(device_ext.extension_name_as_c_str().unwrap())
            })
        {
            device_extension_names.push(ash::khr::shader_non_semantic_info::NAME.as_ptr());
        } else if opts.debuggable_shaders {
            opts.debuggable_shaders = false;
            log::warn!("Unable to load debuggable shaders, missing extension");
        }
        let mut queue_family_indices = HashSet::with_capacity(3);
        queue_family_indices.insert(present_queue_family_index);
        queue_family_indices.insert(graphics_queue_family_index);
        queue_family_indices.insert(transfer_queue_family_index);
        let queue_priorities = &[1f32; 30];
        let queue_create_infos: Vec<_> = queue_family_indices
            .iter()
            .map(|qfi| {
                let queue_family_info =
                    unsafe { instance.get_relevant_physical_device_properties(physical_device) }
                        .queue_families[*qfi as usize];
                let queue_count = queue_priorities
                    .len()
                    .min(queue_family_info.queue_count as usize);
                DeviceQueueCreateInfo::default()
                    .queue_family_index(*qfi)
                    .queue_priorities(&queue_priorities[..queue_count])
            })
            .collect();

        log::trace!("Queue Family Create Infos: {:?}", queue_create_infos);
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
                Device::new(
                    &instance,
                    scored_phys_dev.phys_dev,
                    &dev_ci,
                    debug_string!(true, "Main Device"),
                )
            }
            .map_err(|_| DeviceCreation)?,
        );

        log::trace!("Device info: {:?}", device);

        let shader_compiler = shaderc::Compiler::new().ok_or(ShaderCompilerCreation)?;

        let vert_shader_path = Path::new("shaders/shader.vert");
        let frag_shader_path = Path::new("shaders/shader.frag");

        let (vert_shader_mod, frag_shader_mod) = if !opts.debuggable_shaders {
            let vert_shader_mod = ShaderModule::from_source(
                &device,
                &shader_compiler,
                vert_shader_path,
                ShaderStageFlags::VERTEX,
                "main",
                None,
                debug_string!(device.is_debug(), "Vertex Shader Module"),
            );

            let frag_shader_mod = ShaderModule::from_source(
                &device,
                &shader_compiler,
                frag_shader_path,
                ShaderStageFlags::FRAGMENT,
                "main",
                None,
                debug_string!(device.is_debug(), "Fragment Shader Module"),
            );
            (vert_shader_mod, frag_shader_mod)
        } else {
            let vert_shader_path = PathBuf::from(format!(
                "{}.with_debug_info.spv",
                vert_shader_path.to_str().unwrap()
            ));
            let vert_shader = ShaderModule::from_spirv(
                &device,
                &vert_shader_path,
                ShaderStageFlags::VERTEX,
                "main",
                debug_string!(device.is_debug(), "Vertex Shader Module"),
            );

            let frag_shader_path = PathBuf::from(format!(
                "{}.with_debug_info.spv",
                frag_shader_path.to_str().unwrap()
            ));
            let frag_shader = ShaderModule::from_spirv(
                &device,
                &frag_shader_path,
                ShaderStageFlags::FRAGMENT,
                "main",
                debug_string!(device.is_debug(), "Fragment Shader Module"),
            );

            (vert_shader, frag_shader)
        };
        let (vert_shader_mod, frag_shader_mod) = match (vert_shader_mod, frag_shader_mod) {
            (Ok(mod1), Ok(mod2)) => Ok((mod1, mod2)),
            (Err(e), Ok(_)) | (Ok(_), Err(e)) => Err(ShaderModuleCreationErrors(vec![e])),
            (Err(e1), Err(e2)) => Err(ShaderModuleCreationErrors(vec![e1, e2])),
        }?;
        let descriptor_bindings = &[
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::VERTEX)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER),
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::FRAGMENT),
        ];
        let descriptor_set_layout_ci =
            DescriptorSetLayoutCreateInfo::default().bindings(descriptor_bindings);
        //SAFETY: Valid ci
        let descriptor_set_layout = unsafe {
            DescriptorSetLayout::new(
                &device,
                &descriptor_set_layout_ci,
                debug_string!(device.is_debug(), "Descriptor Set Layout"),
            )
        }
        .map_err(|_| ContextCreationError::DescriptorSetCreation)?;

        let descriptor_set_layouts = &[descriptor_set_layout.as_inner()];

        let pool_sizes = &[
            DescriptorPoolSize::default()
                .descriptor_count(MAX_FRAMES_IN_FLIGHT)
                .ty(DescriptorType::UNIFORM_BUFFER),
            DescriptorPoolSize::default()
                .descriptor_count(MAX_FRAMES_IN_FLIGHT)
                .ty(DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];
        let max_sets = pool_sizes
            .iter()
            .fold(0, |acc, dps| dps.descriptor_count + acc);
        let descriptor_pool_ci = DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);
        let descriptor_pool = Rc::new(
            unsafe {
                DescriptorPool::new(
                    &device,
                    &descriptor_pool_ci,
                    debug_string!(device.is_debug(), "Descriptor Pool"),
                )
            }
            .map_err(|_| DescriptorSetCreation)?,
        );

        let duped_layouts = vec![descriptor_set_layout.as_inner(); MAX_FRAMES_IN_FLIGHT as usize];
        let descriptor_set_ai = DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.as_inner())
            .set_layouts(&duped_layouts);

        //SAFETY: valid ai
        let mut descriptor_sets = unsafe {
            DescriptorSet::alloc(
                &descriptor_pool,
                &descriptor_set_ai,
                Some(|i, _| Some(format!("Descriptor set {}", i))),
            )
        }
        .unwrap();

        let vert_push_constant_range = PushConstantRange::default()
            .stage_flags(ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4<f32>>() as u32);

        let push_constant_ranges = &[vert_push_constant_range];

        let pipeline_layout_ci = PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            PipelineLayout::new(
                &device,
                &pipeline_layout_ci,
                debug_string!(device.is_debug(), "Pipeline layout"),
            )
        }
        .map_err(|e| UnknownVulkan("creating pipeline layout".into(), e))?;
        let graphics_command_pool_ci = CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let graphics_command_pool = Arc::new(
            unsafe {
                CommandPool::new(
                    &device,
                    &graphics_command_pool_ci,
                    debug_string!(device.is_debug(), "Render command pool"),
                )
            }
            .map_err(|_| CommandBufferCreation)?,
        );

        let command_buffers = graphics_command_pool
            .alloc_command_buffers(
                MAX_FRAMES_IN_FLIGHT,
                CommandBufferLevel::PRIMARY,
                Some(|i, _| Some(format!("Main command buffer {}", i))),
            )
            .map_err(|_| CommandBufferCreation)?;

        let mut vertex_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut index_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut render_complete_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        let mut prev_render_complete_fence = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            vertex_buffers.push(
                MappableBuffer::new(
                    &device,
                    size_of_val(VERTICES) as u64,
                    BufferUsageFlags::VERTEX_BUFFER,
                    MemoryUsage::AutoPreferDevice,
                    None,
                    debug_string!(device.is_debug(), "Vertex Buffer {}", i),
                )
                .unwrap(),
            );
            index_buffers.push(
                MappableBuffer::new(
                    &device,
                    size_of_val(INDICES) as u64,
                    BufferUsageFlags::INDEX_BUFFER,
                    MemoryUsage::AutoPreferDevice,
                    None,
                    debug_string!(device.is_debug(), "Index buffer {}", i),
                )
                .unwrap(),
            );
            uniform_buffers.push(
                MappableBuffer::new(
                    &device,
                    size_of::<UniformBuffer>() as u64,
                    BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryUsage::AutoPreferHost,
                    None,
                    debug_string!(device.is_debug(), "uniform buffer {}", i),
                )
                .unwrap(),
            );
            image_available_semaphores.push(
                Semaphore::new(
                    &device,
                    debug_string!(device.is_debug(), "image_available_semaphore [{}]", i,),
                )
                .unwrap(),
            );
            render_complete_semaphores.push(
                Semaphore::new(
                    &device,
                    debug_string!(device.is_debug(), "render_complete_semaphore [{}]", i,),
                )
                .unwrap(),
            );
            prev_render_complete_fence.push(
                Fence::new(
                    &device,
                    true,
                    debug_string!(device.is_debug(), "prev_render_complete_fence [{}]", i,),
                )
                .unwrap(),
            );
        }

        let transfer_command_pool_ci = CommandPoolCreateInfo::default()
            .queue_family_index(transfer_queue_family_index)
            .flags(CommandPoolCreateFlags::TRANSIENT);
        //SAFETY: valid cis
        let transfer_command_pool = Arc::new(
            unsafe {
                CommandPool::new(
                    &device,
                    &transfer_command_pool_ci,
                    debug_string!(device.is_debug(), "Transfer Command Pool"),
                )
            }
            .unwrap(),
        );
        let (gpu_image, gpu_waitable) = GpuImage::from_file(
            "res/texture.png",
            &device,
            &transfer_command_pool,
            &graphics_command_pool,
            transfer_queue_family_index,
            graphics_queue_family_index,
        )
        .unwrap();
        let image_subresource_range = ImageSubresourceRange::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count((gpu_image.mip_levels - 1).max(1))
            .base_array_layer(0)
            .layer_count(1);
        let gpu_image = Arc::new(gpu_image);
        let image_view_ci = ImageViewCreateInfo::default()
            .image(gpu_image.inner)
            .view_type(ImageViewType::TYPE_2D)
            .format(Format::R8G8B8A8_SRGB)
            .subresource_range(image_subresource_range);

        let gpu_image_view = unsafe {
            GpuImageView::new(
                &gpu_image,
                &image_view_ci,
                debug_string!(device.is_debug(), "Texture image view"),
            )
        }
        .unwrap();
        let sampler_ci = SamplerCreateInfo::default()
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR)
            .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(16.)
            .unnormalized_coordinates(false)
            .border_color(BorderColor::INT_OPAQUE_BLACK)
            .compare_enable(false)
            .compare_op(CompareOp::ALWAYS)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.0)
            .max_lod(0.0);

        let texture_sampler = unsafe {
            TextureSampler::new(
                &device,
                &sampler_ci,
                debug_string!(device.is_debug(), "Texture sampler"),
            )
        }
        .unwrap();

        for (i, descriptor_set) in descriptor_sets.iter_mut().enumerate() {
            let buffer_infos = &[DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i].get_inner())
                .offset(0)
                .range(size_of::<UniformBuffer>() as u64)];
            let image_infos = &[DescriptorImageInfo::default()
                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gpu_image_view.inner)
                .sampler(texture_sampler.inner)];

            let descriptor_set_configs = &[
                WriteDescriptorSet::default()
                    .dst_set(descriptor_set.get_inner())
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(buffer_infos),
                WriteDescriptorSet::default()
                    .dst_set(descriptor_set.get_inner())
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(image_infos),
            ];

            //SAFETY: Valid config
            unsafe {
                device
                    .as_inner_ref()
                    .update_descriptor_sets(descriptor_set_configs, &[])
            };
        }

        let fences: Vec<_> = gpu_waitable.get_fences().collect();

        unsafe {
            device
                .as_inner_ref()
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .unwrap();

        Ok(Context {
            frame_index: 0,
            win,
            command_buffers,
            uniform_descriptor_sets: descriptor_sets,
            _instance: instance,
            surface_derived: Some(SurfaceDerived::new(
                &device,
                Arc::new(surface),
                graphics_queue_family_index,
                present_queue_family_index,
                &[&vert_shader_mod, &frag_shader_mod],
                &pipeline_layout,
                multisample_flag,
                None,
            )?),
            graphics_queue_index: graphics_queue_family_index,
            present_queue_index: present_queue_family_index,
            vertex_buffers,
            uniform_buffers,
            index_buffers,
            image_available_semaphores,
            render_complete_semaphores,
            prev_frame_finished_fences: prev_render_complete_fence,
            pipeline_layout,
            device,
            vert_shader_mod,
            frag_shader_mod,
            start_time: Instant::now(),
            _gpu_image_view: gpu_image_view,
            _texture_sampler: texture_sampler,
            multisample_flag: scored_phys_dev.multisample_flag,
            minimized: false,
        })
    }
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width != 0 && new_size.height != 0 {
            if let Some(ref mut surface_derived) = self.surface_derived {
                let surface = surface_derived.surface.clone();
                //Must wait for the device to be idle before dropping any objects
                //potentially referred to by frames in flight
                self.device.wait_idle().unwrap();

                *surface_derived = SurfaceDerived::new(
                    &self.device,
                    surface,
                    self.graphics_queue_index,
                    self.present_queue_index,
                    &[&self.vert_shader_mod, &self.frag_shader_mod],
                    &self.pipeline_layout,
                    self.multisample_flag,
                    Some(&surface_derived.swapchain),
                )
                .unwrap();
                self.minimized = false;
            }
        } else {
            self.minimized = true;
        }
    }
    pub fn draw(&mut self) {
        if !self.minimized {
            match &mut self.surface_derived {
                Some(sd) => {
                    let frame_index = self.frame_index;
                    self.frame_index += 1;
                    let sync_index = frame_index % MAX_FRAMES_IN_FLIGHT as usize;
                    let guard_fence = &mut self.prev_frame_finished_fences[sync_index];
                    guard_fence.wait_and_reset().unwrap();
                    let image_available_semaphore =
                        &mut self.image_available_semaphores[sync_index];
                    let render_complete_semaphore =
                        &mut self.render_complete_semaphores[sync_index];
                    let vertex_buffer = &mut self.vertex_buffers[sync_index];
                    let index_buffer = &mut self.index_buffers[sync_index];
                    let uniform_buffer = &mut self.uniform_buffers[sync_index];
                    let uniform_descriptor_set = &mut self.uniform_descriptor_sets[sync_index];
                    let cb = &mut self.command_buffers[sync_index];
                    //SAFETY: Semaphore derives from same device as swapchain
                    let fb_index = unsafe {
                        sd.swapchain
                            .acquire_next_image(Some(image_available_semaphore), None)
                    }
                    .unwrap()
                    .0;
                    let fb = &mut sd.swapchain_framebuffers[fb_index as usize];
                    let model = Mat4::rotation_z(
                        90f32.to_radians() * self.start_time.elapsed().as_secs_f32(),
                    );
                    let view: Mat4<f32> = Mat4::look_at_rh(
                        Vec3::new(0., -1., 2.),
                        Vec3::broadcast(0.),
                        Vec3::unit_z(),
                    );
                    let vulkan_correction_matrix = Mat4::from_diagonal(Vec4::new(1., -1., 1., 1.));

                    let proj: Mat4<f32> = Mat4::perspective_rh_zo(
                        45f32.to_radians(),
                        sd.swapchain.get_aspect_ratio(),
                        0.01,
                        1000000.0,
                    ) * vulkan_correction_matrix;

                    vertex_buffer.upload_data(VERTICES);
                    index_buffer.upload_data(INDICES);
                    uniform_buffer.upload_data(&[UniformBuffer { view, proj }]);

                    cb.record_and_submit(
                        self.graphics_queue_index,
                        0,
                        &[image_available_semaphore.get_inner()],
                        &[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                        &[render_complete_semaphore.get_inner()],
                        &CommandBufferBeginInfo::default()
                            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        guard_fence.get_inner(),
                        |dev, cb| -> Result<(), ash::vk::Result> {
                            //SAFETY: We know dev and cb are linked. All other vars
                            //used are derived from device
                            unsafe {
                                if let Some(debug_device) = self.device.debug_device_ref() {
                                    let label_name = CString::new(format!(
                                        "Rendering frame {} si {} to fb {}",
                                        frame_index, sync_index, fb_index
                                    ))
                                    .unwrap();

                                    let debug_label = DebugUtilsLabelEXT::default()
                                        .label_name(&label_name)
                                        .color(LIME);

                                    debug_device.cmd_begin_debug_utils_label(cb, &debug_label);
                                }
                                dev.cmd_push_constants(
                                    cb,
                                    self.pipeline_layout.get_inner(),
                                    ShaderStageFlags::VERTEX,
                                    0,
                                    bytemuck::cast_slice(&[model]),
                                );

                                dev.cmd_bind_descriptor_sets(
                                    cb,
                                    PipelineBindPoint::GRAPHICS,
                                    self.pipeline_layout.get_inner(),
                                    0,
                                    &[uniform_descriptor_set.get_inner()],
                                    &[],
                                );
                                dev.cmd_bind_index_buffer(
                                    cb,
                                    index_buffer.get_inner(),
                                    0,
                                    IndexType::UINT16,
                                );
                                dev.cmd_bind_vertex_buffers(
                                    cb,
                                    0,
                                    &[vertex_buffer.get_inner()],
                                    &[0],
                                );

                                dev.cmd_begin_render_pass(
                                    cb,
                                    &RenderPassBeginInfo::default()
                                        .clear_values(&[
                                            ClearValue {
                                                color: ClearColorValue {
                                                    float32: [0.0, 0.0, 0.0, 1.0],
                                                },
                                            },
                                            ClearValue {
                                                depth_stencil: ClearDepthStencilValue {
                                                    depth: 1.0,
                                                    stencil: 0,
                                                },
                                            },
                                        ])
                                        .render_area(sd.swapchain.as_rect())
                                        .render_pass(sd.render_pass.get_inner())
                                        .framebuffer(fb.get_inner()),
                                    SubpassContents::INLINE,
                                );
                                dev.cmd_bind_pipeline(
                                    cb,
                                    PipelineBindPoint::GRAPHICS,
                                    sd.pipeline.get_inner(),
                                );
                                dev.cmd_draw_indexed(cb, INDICES.len() as u32, 1, 0, 0, 0);
                                dev.cmd_end_render_pass(cb);
                                Ok(())
                            }
                        },
                    )
                    .unwrap();

                    fb.present(
                        self.present_queue_index,
                        &[render_complete_semaphore.get_inner()],
                    )
                    .unwrap();
                }
                None => {}
            }
        }
    }

    pub fn win_id(&self) -> WindowId {
        self.win.id()
    }
}

struct ScoredPhysDev {
    score: u64,
    phys_dev: PhysicalDevice,
    graphics_queue_index: u32,
    present_queue_index: u32,
    transfer_queue_index: u32,
    multisample_flag: SampleCountFlags,
}
impl PartialEq for ScoredPhysDev {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl PartialOrd for ScoredPhysDev {
    fn partial_cmp(&self, other: &Self) -> std::option::Option<std::cmp::Ordering> {
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
    shared_graphics_transfer_queue: bool,
) -> Option<ScoredPhysDev> {
    //SAFETY:phys_dev must be derived from instance
    let properties = unsafe { instance.get_relevant_physical_device_properties(phys_dev) };
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
                .find(|(_, qfp)| qfp.queue_flags.intersects(QueueFlags::GRAPHICS))
        })
        .map(|(i, _)| i as u32);
    let present_queue_index = graphics_queue_index
        .filter(|graphics_queue_index| {
            //SAFETY: qfi is in bounds, phys_dev comes from same instance as surface
            unsafe { surface.does_queue_support_presentation(phys_dev, *graphics_queue_index) }
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
                    unsafe { surface.does_queue_support_presentation(phys_dev, *qfi as u32) }
                })
                .map(|(i, _)| i as u32)
        });

    let transfer_queue_index = shared_graphics_transfer_queue
        .then_some(())
        .and(graphics_queue_index)
        .or_else(|| {
            properties
                .queue_families
                .iter()
                .copied()
                .enumerate()
                .find(|(_qfi, props)| {
                    props.queue_flags.contains(QueueFlags::TRANSFER)
                        && !(props
                            .queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE))
                })
                .map(|(i, _props)| i as u32)
                .or(graphics_queue_index)
        });
    let shared_max_sample_count = {
        let counts = properties.props.limits.framebuffer_color_sample_counts
            & properties.props.limits.framebuffer_depth_sample_counts;
        [
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ]
        .iter()
        .cloned()
        .find(|c| counts.contains(*c))
        .unwrap_or(vk::SampleCountFlags::TYPE_1)
    };
    if properties.extensions.iter().any(|ext| {
        ash::khr::swapchain::NAME.eq(ext
            .extension_name_as_c_str()
            .expect("It'd be weird if we got an extension that wasn't a valid cstr"))
        //SAFETY: phys_dev from same source as surface
    }) && unsafe {
        surface
            .get_compatible_swapchain_info(phys_dev)
            .ok()
            .map_or(false, |swap_info| {
                !swap_info.formats.is_empty() && !swap_info.present_modes.is_empty()
            })
    } {
        graphics_queue_index.and_then(|graphics_queue_index| {
            present_queue_index.map(|present_queue_index| {
                let transfer_queue_index = transfer_queue_index.unwrap();
                //Weight GPUs more highly based on what type they are. If unknown,
                //just return some weird low value but better than CPU
                let device_type_score = match properties.props.device_type {
                    PhysicalDeviceType::DISCRETE_GPU => 1000,
                    PhysicalDeviceType::INTEGRATED_GPU => 500,
                    PhysicalDeviceType::CPU => 0,
                    _ => 100,
                };
                //If multiple of same category, pick the one with a shared graphics and presentation queue
                let shared_pres_graph_queue_score = if graphics_queue_index == present_queue_index {
                    100
                } else {
                    0
                };
                let shared_transfer_queue_score = if transfer_queue_index == graphics_queue_index {
                    0
                } else if transfer_queue_index == present_queue_index {
                    50
                } else {
                    100
                };

                let multisample_score =
                    map_multisample_flag_to_sample_count(shared_max_sample_count);

                let queue_score =
                    shared_pres_graph_queue_score + shared_transfer_queue_score + multisample_score;
                ScoredPhysDev {
                    score: queue_score + device_type_score,
                    phys_dev,
                    graphics_queue_index,
                    present_queue_index,
                    transfer_queue_index,
                    multisample_flag: shared_max_sample_count,
                }
            })
        })
    } else {
        None
    }
}

const CYAN: [f32; 4] = [0., 0.976, 1., 1.0];
const MAGENTA: [f32; 4] = [0.839, 0.067, 0.804, 1.0];
const YELLOW: [f32; 4] = [0.957, 0.98, 0.424, 1.0];
const LIME: [f32; 4] = [0.514, 0.969, 0.333, 1.0];

fn map_multisample_flag_to_sample_count(flag: SampleCountFlags) -> u64 {
    match flag {
        SampleCountFlags::TYPE_1 => 1,
        SampleCountFlags::TYPE_2 => 2,
        SampleCountFlags::TYPE_4 => 4,
        SampleCountFlags::TYPE_8 => 8,
        SampleCountFlags::TYPE_16 => 16,
        SampleCountFlags::TYPE_32 => 32,
        SampleCountFlags::TYPE_64 => 64,
        _ => panic!("You need to ensure you're passing *singular* flags to this function"),
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

#[derive(Debug)]
struct GpuImage {
    inner: ash::vk::Image,
    parent: Arc<Device>,
    allocation: vk_mem::Allocation,
    mip_levels: u32,
}

#[derive(Debug, thiserror::Error)]
enum LoadTextureFromFileError {
    #[error("Attempted to load invalid file")]
    InvalidTextureFile,
    #[error("Error submitting command buffer")]
    GpuUploadError,
    #[error("File does not exist")]
    NoSuchFile,
    #[error("Could not generate mipmaps")]
    MipmapGenerationError,
}
impl GpuImage {
    fn from_file(
        path: impl AsRef<Path> + Clone,
        device: &Arc<Device>,
        transfer_command_pool: &Arc<CommandPool>,
        graphics_command_pool: &Arc<CommandPool>,
        transfer_queue_index: u32,
        graphics_queue_index: u32,
    ) -> Result<(Self, impl FenceProducer), LoadTextureFromFileError> {
        let image_file = std::io::BufReader::new(
            std::fs::File::open(path.clone()).map_err(|_| LoadTextureFromFileError::NoSuchFile)?,
        );
        let image_buffer = image::load(image_file, image::ImageFormat::Png)
            .map_err(|_| LoadTextureFromFileError::InvalidTextureFile)?
            .into_rgba8();

        //SAFETY: Valid ci and ai
        let mut staging_buffer: MappableBuffer<u8> = MappableBuffer::new(
            device,
            size_of_val(image_buffer.as_ref()) as u64,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            None,
            debug_string!(device.is_debug(), "Staging buffer"),
        )
        .unwrap();

        let raw_image = image_buffer.as_raw();

        staging_buffer.upload_data(raw_image.as_ref());

        let mut image_transfer_command_buffer = transfer_command_pool
            .alloc_command_buffer(CommandBufferLevel::PRIMARY)
            .unwrap();

        let mip_levels = log2_floor(image_buffer.width().max(image_buffer.height()));

        let image_create_info = ImageCreateInfo::default()
            .image_type(ImageType::TYPE_2D)
            .extent(Extent3D {
                width: image_buffer.width(),
                height: image_buffer.height(),
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(Format::R8G8B8A8_SRGB)
            .tiling(ImageTiling::OPTIMAL)
            .initial_layout(ImageLayout::UNDEFINED)
            .usage(
                ImageUsageFlags::SAMPLED
                    | ImageUsageFlags::TRANSFER_DST
                    | ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(SharingMode::EXCLUSIVE)
            .samples(SampleCountFlags::TYPE_1);
        let image_allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        //SAFETY: Valid cis
        let gpu_image = unsafe {
            GpuImage::new(
                device,
                &image_create_info,
                &image_allocation_info,
                debug_string!(device.is_debug(), "Texture Image"),
            )
        }
        .unwrap();

        let whole_image_subresource_range = ImageSubresourceRange::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let image_transfer_begin_info =
            CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        //Sets up image to be transfered into from the buffer
        let image_pre_load_barrier = ImageMemoryBarrier::default()
            .old_layout(ImageLayout::UNDEFINED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .image(gpu_image.inner)
            .subresource_range(whole_image_subresource_range)
            .src_access_mask(AccessFlags::empty())
            .dst_access_mask(AccessFlags::TRANSFER_WRITE);

        //Transfers the image to the graphics queue and makes it so that the
        //whole image is in TRANSFER_SRC_OPTIMAL
        let image_post_load_barrier = ImageMemoryBarrier::default()
            .image(gpu_image.inner)
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(transfer_queue_index)
            .dst_queue_family_index(graphics_queue_index)
            .subresource_range(whole_image_subresource_range)
            .src_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(AccessFlags::TRANSFER_READ);

        let image_subresource_layers = ImageSubresourceLayers::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let regions = &[BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(image_subresource_layers)
            .image_offset(Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(Extent3D {
                width: image_buffer.width(),
                height: image_buffer.height(),
                depth: 1,
            })];

        let transfer_complete_semaphore = Semaphore::new(
            device,
            debug_string!(
                device.is_debug(),
                "Texture Transfer Complete Semaphore For {:?}",
                path.clone().as_ref().to_str().unwrap()
            ),
        )
        .unwrap();

        image_transfer_command_buffer
            .record_and_submit(
                transfer_queue_index,
                0,
                &[],
                &[],
                &[transfer_complete_semaphore.get_inner()],
                &image_transfer_begin_info,
                ash::vk::Fence::null(),
                |dev, cb| -> Result<(), ash::vk::Result> {
                    //SAFETY: cb and all parameters are from the same device

                    unsafe {
                        if let Some(debug_device) = device.debug_device_ref() {
                            let debug_label = CString::new(format!(
                                "Transferring image at {:?} to base mip level",
                                path.clone().as_ref()
                            ))
                            .unwrap();
                            let debug_label = DebugUtilsLabelEXT::default()
                                .label_name(&debug_label)
                                .color(MAGENTA);

                            debug_device.cmd_begin_debug_utils_label(cb, &debug_label);
                        }
                        dev.cmd_pipeline_barrier(
                            cb,
                            PipelineStageFlags::TOP_OF_PIPE,
                            PipelineStageFlags::TRANSFER,
                            DependencyFlags::empty(),
                            &[],
                            &[],
                            &[image_pre_load_barrier],
                        );
                        dev.cmd_copy_buffer_to_image(
                            cb,
                            staging_buffer.get_inner(),
                            gpu_image.inner,
                            ImageLayout::TRANSFER_DST_OPTIMAL,
                            regions,
                        );
                        dev.cmd_pipeline_barrier(
                            cb,
                            PipelineStageFlags::TRANSFER,
                            PipelineStageFlags::TRANSFER,
                            DependencyFlags::empty(),
                            &[],
                            &[],
                            &[image_post_load_barrier],
                        );
                    };

                    if let Some(debug_device) = device.debug_device_ref() {
                        unsafe { debug_device.cmd_end_debug_utils_label(cb) };
                    }

                    Ok(())
                },
            )
            .map_err(|_| LoadTextureFromFileError::GpuUploadError)?;
        let generate_mipmaps_begin_info =
            CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let mut mipmap_generate_command_buffer = graphics_command_pool
            .alloc_command_buffer(CommandBufferLevel::PRIMARY)
            .unwrap();

        let texture_ready_fence = Fence::new(
            device,
            false,
            debug_string!(device.is_debug(), "Texture Ready Fence"),
        )
        .unwrap();

        mipmap_generate_command_buffer
            .record_and_submit(
                graphics_queue_index,
                0,
                &[transfer_complete_semaphore.get_inner()],
                &[PipelineStageFlags::TRANSFER],
                &[],
                &generate_mipmaps_begin_info,
                texture_ready_fence.get_inner(),
                |dev, cb| -> Result<(), ash::vk::Result> {
                    let acquire_image_barrier = ImageMemoryBarrier::default()
                        .src_queue_family_index(transfer_queue_index)
                        .dst_queue_family_index(graphics_queue_index)
                        .image(gpu_image.inner)
                        .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .subresource_range(whole_image_subresource_range);

                    if transfer_queue_index != graphics_queue_index {
                        unsafe {
                            dev.cmd_pipeline_barrier(
                                cb,
                                PipelineStageFlags::TRANSFER,
                                PipelineStageFlags::TRANSFER,
                                DependencyFlags::empty(),
                                &[],
                                &[],
                                &[acquire_image_barrier],
                            );
                        }
                    }
                    let mut prev_level_preblit_barrier = None;
                    let mut image_barriers = Vec::with_capacity(2);
                    if let Some(debug_device) = device.debug_device_ref() {
                        let label_name = CString::new(format!(
                            "Generating mipmaps for texture {:?}",
                            path.clone().as_ref()
                        ))
                        .unwrap();
                        let debug_label = DebugUtilsLabelEXT::default()
                            .label_name(&label_name)
                            .color(YELLOW);

                        unsafe { debug_device.cmd_begin_debug_utils_label(cb, &debug_label) };
                    }
                    for mip_level in 1..mip_levels {
                        if let Some(debug_device) = device.debug_device_ref() {
                            let label_name = CString::new(format!(
                                "Mip Level {} of {:?}",
                                mip_level,
                                path.clone().as_ref()
                            ))
                            .unwrap();

                            let debug_label = DebugUtilsLabelEXT::default()
                                .label_name(&label_name)
                                .color(CYAN);
                            unsafe { debug_device.cmd_begin_debug_utils_label(cb, &debug_label) };
                        }
                        let prev_level = mip_level - 1;

                        let current_level_range = ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .level_count(1)
                            .base_mip_level(mip_level);

                        let current_layer_preblit_barrier = ImageMemoryBarrier::default()
                            .image(gpu_image.inner)
                            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                            .old_layout(ImageLayout::UNDEFINED)
                            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_access_mask(AccessFlags::empty())
                            .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                            .subresource_range(current_level_range);

                        image_barriers.clear();
                        image_barriers.push(current_layer_preblit_barrier);
                        if let Some(b) = prev_level_preblit_barrier {
                            image_barriers.push(b)
                        }

                        let prev_level_layer = ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .mip_level(mip_level - 1)
                            .base_array_layer(0)
                            .layer_count(1);
                        let mip_level_layer = ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .mip_level(mip_level)
                            .base_array_layer(0)
                            .layer_count(1);
                        let prev_level_width = image_buffer.width() / 2u32.pow(prev_level);
                        let prev_level_height = image_buffer.height() / 2u32.pow(prev_level);

                        let mip_level_height =
                            (image_buffer.height() / 2_u32.pow(mip_level)).max(1);
                        let mip_level_width = (image_buffer.width() / 2u32.pow(mip_level)).max(1);
                        let blit_info = ImageBlit::default()
                            .src_offsets([
                                Offset3D { x: 0, y: 0, z: 0 },
                                Offset3D {
                                    x: prev_level_width as i32,
                                    y: prev_level_height as i32,
                                    z: 1,
                                },
                            ])
                            .src_subresource(prev_level_layer)
                            .dst_offsets([
                                Offset3D { x: 0, y: 0, z: 0 },
                                Offset3D {
                                    x: mip_level_width as i32,
                                    y: mip_level_height as i32,
                                    z: 1,
                                },
                            ])
                            .dst_subresource(mip_level_layer);
                        unsafe {
                            dev.cmd_pipeline_barrier(
                                cb,
                                PipelineStageFlags::TRANSFER,
                                PipelineStageFlags::TRANSFER,
                                DependencyFlags::empty(),
                                &[],
                                &[],
                                &image_barriers,
                            );
                            dev.cmd_blit_image(
                                cb,
                                gpu_image.inner,
                                ImageLayout::TRANSFER_SRC_OPTIMAL,
                                gpu_image.inner,
                                ImageLayout::TRANSFER_DST_OPTIMAL,
                                &[blit_info],
                                vk::Filter::LINEAR,
                            );
                        }
                        //Transfer the level we just wrote into
                        //TRANSFER_SRC_OPTIMAL for future blits
                        prev_level_preblit_barrier = Some(
                            current_layer_preblit_barrier
                                .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
                                .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL),
                        );

                        if let Some(debug_device) = device.debug_device_ref() {
                            unsafe { debug_device.cmd_end_debug_utils_label(cb) };
                        }
                    }

                    let whole_image_but_last_level_range = ImageSubresourceRange::default()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        //If there's only one mip level, prev_image_barrier
                        //will be None. Otherwise, that barrier will handle
                        //that mip level
                        .level_count((mip_levels - 1).max(1));

                    let final_image_barrier = ImageMemoryBarrier::default()
                        .src_access_mask(AccessFlags::TRANSFER_READ)
                        .dst_access_mask(AccessFlags::SHADER_READ)
                        .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .subresource_range(whole_image_but_last_level_range)
                        .image(gpu_image.inner);
                    image_barriers.clear();
                    if let Some(b) = prev_level_preblit_barrier {
                        image_barriers.push(
                            b.new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .dst_access_mask(AccessFlags::SHADER_READ),
                        )
                    }
                    image_barriers.push(final_image_barrier);

                    unsafe {
                        dev.cmd_pipeline_barrier(
                            cb,
                            PipelineStageFlags::TRANSFER,
                            PipelineStageFlags::FRAGMENT_SHADER,
                            DependencyFlags::empty(),
                            &[],
                            &[],
                            &image_barriers,
                        );
                        if let Some(debug_device) = device.debug_device_ref() {
                            debug_device.cmd_end_debug_utils_label(cb);
                        }
                    }
                    Ok(())
                },
            )
            .map_err(|_| LoadTextureFromFileError::MipmapGenerationError)?;
        struct Return {
            _staging_buffer: MappableBuffer<u8>,
            _cb: CommandBuffer,
            _sem: Semaphore,
            fence: Fence,
            _cb2: CommandBuffer,
        }

        impl FenceProducer for Return {
            type Iter = std::iter::Once<ash::vk::Fence>;
            fn get_fences(&self) -> Self::Iter {
                std::iter::once(self.fence.get_inner())
            }
        }

        Ok((
            gpu_image,
            Return {
                _staging_buffer: staging_buffer,
                _cb: image_transfer_command_buffer,
                _cb2: mipmap_generate_command_buffer,
                _sem: transfer_complete_semaphore,
                fence: texture_ready_fence,
            },
        ))
    }
    //SAFETY REQUIREMENTS: Valid ci and ai
    unsafe fn new(
        device: &Arc<Device>,
        image_create_info: &ImageCreateInfo,
        image_allocation_info: &vk_mem::AllocationCreateInfo,
        debug_string: Option<String>,
    ) -> VkResult<Self> {
        //SAFETY: valid cis
        let (inner, allocation) = unsafe {
            device
                .get_allocator_ref()
                .create_image(image_create_info, image_allocation_info)
        }?;

        associate_debug_name!(device, inner, debug_string);

        Ok(Self {
            parent: device.clone(),
            inner,
            allocation,

            mip_levels: image_create_info.mip_levels,
        })
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        //SAFETY:
        unsafe {
            self.parent
                .get_allocator_ref()
                .destroy_image(self.inner, &mut self.allocation)
        };
    }
}

#[derive(Debug)]

struct GpuImageView {
    parent: Arc<GpuImage>,
    inner: ImageView,
}

impl Drop for GpuImageView {
    fn drop(&mut self) {
        unsafe {
            self.parent
                .parent
                .as_inner_ref()
                .destroy_image_view(self.inner, None)
        };
    }
}

impl GpuImageView {
    unsafe fn new(
        parent_image: &Arc<GpuImage>,
        ci: &ImageViewCreateInfo,
        debug_string: Option<String>,
    ) -> VkResult<Self> {
        let inner = unsafe {
            parent_image
                .parent
                .as_inner_ref()
                .create_image_view(ci, None)
        }?;

        associate_debug_name!(parent_image.parent, inner, debug_string);

        Ok(Self {
            parent: parent_image.clone(),
            inner,
        })
    }
}

#[derive(Debug)]
struct TextureSampler {
    parent: Arc<Device>,
    inner: Sampler,
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        unsafe { self.parent.as_inner_ref().destroy_sampler(self.inner, None) };
    }
}

impl TextureSampler {
    unsafe fn new(
        device: &Arc<Device>,
        sampler_ci: &SamplerCreateInfo,
        debug_string: Option<String>,
    ) -> VkResult<Self> {
        let inner = unsafe { device.as_inner_ref().create_sampler(sampler_ci, None) }?;
        associate_debug_name!(device, inner, debug_string);
        Ok(Self {
            parent: device.clone(),
            inner,
        })
    }
}

trait Mat4Ext<T> {
    fn from_diagonal(v: Vec4<T>) -> Mat4<T>;
}

impl<T: Zero + One + MulAssign> Mat4Ext<T> for Mat4<T> {
    fn from_diagonal(v: Vec4<T>) -> Mat4<T> {
        let mut mat = Mat4::identity();
        mat[(0, 0)] *= v.x;
        mat[(1, 1)] *= v.y;
        mat[(2, 2)] *= v.z;
        mat[(3, 3)] *= v.w;
        mat
    }
}
