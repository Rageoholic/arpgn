//FILE SAFETY INVARIANTS
// * Do not expose ShaderModule::inner

// * Only create in new and only destroy ShaderModule::inner in drop

// * Do not destroy if you get an inner ref

// * Do not destroy except in drop

use std::{
    ffi::{CStr, CString},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use ash::vk::{
    Result as RawVkResult, ShaderModule as RawShaderModule,
    ShaderModuleCreateInfo, ShaderStageFlags,
};
use shaderc::ShaderKind;

use super::Device;

#[derive(Debug)]

pub enum Error {
    FileLoad(PathBuf, std::io::Error),
    ShaderCompilation(shaderc::Error),
    MemoryExhaustion,
}

#[derive(Debug)]
pub struct ShaderModule {
    inner: RawShaderModule,

    parent: Arc<Device>,
    ty: ShaderStageFlags,
    name: CString,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        //SAFETY: The shader module was created with this device. Known because
        //we only let you create this struct in new and new ensures that this is
        //the case.
        unsafe {
            self.parent
                .as_inner_ref()
                .destroy_shader_module(self.inner, None)
        };
    }
}

impl ShaderModule {
    pub fn new(
        device: &Arc<Device>,
        shader_compiler: &shaderc::Compiler,
        shader_path: &Path,
        ty: ShaderStageFlags,
        entry: &str,
        options: Option<&shaderc::CompileOptions>,
    ) -> Result<Self, Error> {
        use Error::*;
        let mut file = std::fs::File::open(shader_path)
            .map_err(|err| Error::FileLoad(shader_path.into(), err))?;

        let source = {
            let mut source = String::new();
            file.read_to_string(&mut source)
                .map_err(|err| Error::FileLoad(shader_path.into(), err))?;
            Ok(source)
        }?;

        let spirv = shader_compiler
            .compile_into_spirv(
                &source,
                Self::shader_stage_flags_to_kind(ty),
                shader_path.to_str().unwrap(),
                entry,
                options,
            )
            .map_err(|err| ShaderCompilation(err))?;

        let inner = {
            let ci = ShaderModuleCreateInfo::default().code(spirv.as_binary());
            //SAFETY: Passing valid code to the ci. Shaderc's job
            unsafe { device.as_inner_ref().create_shader_module(&ci, None) }
                .map_err(|err| match err {
                    RawVkResult::ERROR_OUT_OF_HOST_MEMORY
                    | RawVkResult::ERROR_OUT_OF_DEVICE_MEMORY => {
                        MemoryExhaustion
                    }
                    _ => unreachable!(),
                })
        }?;
        Ok(Self {
            inner,
            parent: device.clone(),
            ty,
            name: CString::new(entry).unwrap(),
        })
    }

    fn shader_stage_flags_to_kind(flags: ShaderStageFlags) -> ShaderKind {
        use ShaderKind::*;
        if flags.intersects(ShaderStageFlags::VERTEX) {
            Vertex
        } else if flags.intersects(ShaderStageFlags::FRAGMENT) {
            Fragment
        } else {
            todo!()
        }
    }

    pub(crate) fn as_raw(&self) -> RawShaderModule {
        self.inner
    }
    pub fn get_stage(&self) -> ShaderStageFlags {
        self.ty
    }
    pub fn get_name(&self) -> &CStr {
        &self.name
    }
}
