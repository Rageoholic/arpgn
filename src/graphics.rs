use std::{
    collections::HashSet, ffi::CStr, fmt::Debug, str::FromStr, sync::Arc,
};

use ash::{vk, LoadingError};
use debug_messenger::DebugMessenger;
use device::Device;
use instance::Instance;
use structopt::StructOpt;
use strum::EnumString;

use surface::Surface;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::ActiveEventLoop,
    raw_window_handle::HasDisplayHandle,
    window::{Window, WindowAttributes, WindowId},
};

const DEFAULT_WINDOW_WIDTH: u32 = 1280;

const DEFAULT_WINDOW_HEIGHT: u32 = 720;

mod debug_messenger;
mod device;
mod instance;
mod surface;

const _MAX_FRAMES_IN_FLIGHT: usize = 2;

struct ContextNonDebug {
    _entry: Arc<ash::Entry>,
    _instance: Arc<Instance>,
    _debug_messenger: Option<DebugMessenger>,
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
    win: Arc<Window>,
    _nd: ContextNonDebug,
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

#[derive(Debug)]
pub enum Error {
    #[allow(dead_code)]
    Loading(LoadingError),
    InstanceCreation,
    #[allow(dead_code)]
    _MissingMandatoryExtensions(Vec<String>),
    NoSuitableDevice,
    _DeviceCreation,
    SurfaceCreation,
    _SwapchainCreation,
    _CommandBufferCreation,
    #[allow(dead_code)]
    Unknown(String),
    #[allow(dead_code)]
    WindowCreation(winit::error::OsError),
}

const KHRONOS_VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

impl Context {
    //TODO: Split this function the fuck up. Christ this is long
    pub fn new(
        event_loop: &ActiveEventLoop,
        mut opts: ContextCreateOpts,
    ) -> Result<Self, Error> {
        let entry =
        //SAFETY: Must be dropped *after* instance, which we accomplish by
        //having instance hold on to a ref counted pointer to entry
            Arc::new(unsafe { ash::Entry::load().map_err(Error::Loading) }?);
        //SAFETY: Should be good?
        let vk_version = match unsafe { entry.try_enumerate_instance_version() }
        {
            Ok(ver) => ver.unwrap_or(vk::API_VERSION_1_0),

            Err(_) => unreachable!(),
        };
        log::info!(
            "Available vk version {}.{}.{}",
            vk::api_version_major(vk_version),
            vk::api_version_minor(vk_version),
            vk::api_version_patch(vk_version)
        );
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk_version)
            .application_name(c"arpgn");
        let avail_extensions =
        //SAFETY: Should always be good
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|_| {
                    Error::Unknown(String::from_str(
                        "Couldn't load instance extensions",
                    ).unwrap())
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
            return Err(Error::_MissingMandatoryExtensions(
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

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_exts)
            .enabled_layer_names(&layer_names);

        let all_debug_message_types =
            vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

        let mut debug_utils_messenger_ci =
            vk::DebugUtilsMessengerCreateInfoEXT::default()
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
                .map_err(|_| Error::InstanceCreation)?,
        );

        let debug_messenger = if opts.graphics_validation_layers
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
                .map_err(Error::WindowCreation)?,
        );

        let surface = Arc::new(
            surface::Surface::new(&instance, &win)
                .map_err(|_| Error::SurfaceCreation)?,
        );

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
            .ok_or(Error::NoSuitableDevice)?;
        //TODO: Properly check for these extensions
        let device_extension_names = vec![ash::khr::swapchain::NAME.as_ptr()];
        let mut queue_family_indices = HashSet::with_capacity(2);
        queue_family_indices.insert(scored_phys_dev.present_queue_index);
        queue_family_indices.insert(scored_phys_dev.graphics_queue_index);
        let queue_create_infos: Vec<_> = queue_family_indices
            .iter()
            .map(|qfi| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*qfi)
                    .queue_priorities(&[1.0])
            })
            .collect();
        #[allow(deprecated)]
        let dev_ci = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&device_extension_names)
            .queue_create_infos(&queue_create_infos)
            //We're using this to be compatible with older implementations. This
            //isn't necessary but it's like. 2 values in the struct that would
            //have to be zeroed anyways. Who cares.
            .enabled_layer_names(&layer_names);
        //SAFETY: valid ci and phys_dev is derived from instance. We accomplished these
        let _device = unsafe {
            Device::new(&instance, scored_phys_dev.phys_dev, &dev_ci)
        };

        Ok(Context {
            win,
            _nd: ContextNonDebug {
                _entry: entry,
                _instance: instance,
                _debug_messenger: debug_messenger,
            },
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
    phys_dev: vk::PhysicalDevice,
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
    phys_dev: vk::PhysicalDevice,
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
            qfp.queue_flags.intersects(vk::QueueFlags::GRAPHICS)
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
                    qfp.queue_flags.intersects(vk::QueueFlags::GRAPHICS)
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
    graphics_queue_index.and_then(|graphics_queue_index| {
        present_queue_index.map(|present_queue_index| {
            //Weight CPUs more highly based on what type they are. If unknown,
            //just return some weird low value but better than CPU
            let device_type_score = match properties.props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
                vk::PhysicalDeviceType::CPU => 0,
                _ => 100,
            };
            //If multiple of same category, pick the one with a shared graphics and presentation queue
            let queue_score = if graphics_queue_index == present_queue_index {
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
        ValidationLevel::Warn => warning,
        ValidationLevel::Info => info,
        ValidationLevel::Verbose => verbose,
    }
}
