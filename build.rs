use std::env;
use std::fs::read_dir;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;

use copy_to_output::copy_to_output;

enum ShaderType {
    Frag,
    Vert,
    Support,
}

enum TryFromShaderTypeError {
    UnrecognizedFileType,
}

impl TryFrom<&str> for ShaderType {
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.eq("vert") {
            Ok(ShaderType::Vert)
        } else if value.eq("frag") {
            Ok(ShaderType::Frag)
        } else if value.eq("glsl") {
            Ok(ShaderType::Support)
        } else {
            Err(Self::Error::UnrecognizedFileType)
        }
    }

    type Error = TryFromShaderTypeError;
}

fn main() {
    println!("cargo::rerun-if-changed=res/*");
    println!("cargo::rerun-if-changed=shaders/*");
    copy_to_output("res", &env::var("PROFILE").unwrap()).expect("Could not copy");

    copy_to_output("shaders", &env::var("PROFILE").unwrap()).expect("Unable to copy shaders");

    let target_dir = env::var("OUT_DIR").unwrap();
    let out_dir = PathBuf::from(target_dir);

    let shader_dir = read_dir("shaders").unwrap();
    let mut shader_target_dir = out_dir.clone();
    shader_target_dir.push("shaders");
    eprintln!("{:?}", shader_target_dir);

    let shader_target_dir = shader_target_dir;

    if let Err(e) = std::fs::create_dir(shader_target_dir) {
        if e.kind() != ErrorKind::AlreadyExists {
            println!("cargo::warning=COULD NOT CREATE SHADER DIRECTORY IN OUT")
        }
    }

    for entry in shader_dir {
        let entry = entry.unwrap();

        let path = entry.path();

        if let Ok(shader_type) = path
            .extension()
            .ok_or(TryFromShaderTypeError::UnrecognizedFileType)
            .and_then(|os_str| {
                os_str
                    .to_str()
                    .ok_or(TryFromShaderTypeError::UnrecognizedFileType)
                    .and_then(TryInto::<ShaderType>::try_into)
            })
        {
            match shader_type {
                ShaderType::Frag | ShaderType::Vert => {
                    //Currently a hack, makes the .spv files in tree. Not ideal
                    let mut target_dir = PathBuf::from("shaders");
                    let artifact_name = format!(
                        "{}.with_debug_info.spv",
                        entry.file_name().to_str().unwrap()
                    );

                    target_dir.push(&artifact_name);
                    Command::new("glslangValidator")
                        .args([
                            "-e",
                            "main",
                            "-gVS",
                            "-V",
                            "-o",
                            target_dir.to_str().unwrap(),
                            path.as_os_str().to_str().unwrap(),
                        ])
                        .output()
                        .expect("failed to compile shader");
                }
                ShaderType::Support => todo!(),
            }
        }
    }
}
