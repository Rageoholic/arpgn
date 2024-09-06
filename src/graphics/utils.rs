macro_rules! associate_debug_name {
    ($device:expr, $inner: expr,$name:expr) => {
        if $device.is_debug() {
            if let Some(debug_name) = $name {
                let debug_name = std::ffi::CString::new(debug_name).unwrap();
                let name_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle($inner)
                    .object_name(&debug_name);
                $device.associate_debug_name(&name_info);
            }
        }
    };
}

pub(crate) use associate_debug_name;
