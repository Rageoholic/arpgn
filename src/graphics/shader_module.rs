//FILE SAFETY INVARIANTS
// * Do not expose ShaderModule::inner
// * Only create in new and only destroy ShaderModule::inner in drop

use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use ash::vk;

#[derive(Debug)]

pub enum Error {
    FileLoad(PathBuf, std::io::Error),
    ShaderCompilation(shaderc::Error),
    MemoryExhaustion,
}

#[allow(dead_code)]
pub struct ShaderModule {
    inner: vk::ShaderModule,
    parent: Arc<super::Device>,
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
        device: &Arc<super::Device>,
        shader_compiler: &shaderc::Compiler,
        shader_path: &Path,
        ty: shaderc::ShaderKind,
        entry: &str,
        options: Option<&shaderc::CompileOptions>,
    ) -> Result<Self, Error> {
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
                ty,
                shader_path.to_str().unwrap(),
                entry,
                options,
            )
            .map_err(|err| Error::ShaderCompilation(err))?;

        let inner = {
            let ci =
                vk::ShaderModuleCreateInfo::default().code(spirv.as_binary());
            //SAFETY: Passing valid code to the ci. Shaderc's job
            unsafe { device.as_inner_ref().create_shader_module(&ci, None) }
                .map_err(|err| match err {
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY
                    | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                        Error::MemoryExhaustion
                    }
                    _ => unreachable!(),
                })
        }?;
        Ok(Self {
            inner,
            parent: device.clone(),
        })
    }
}
