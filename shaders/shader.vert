#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location=2) in vec2 inCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 texCoord;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view; mat4 proj;
} uniforms;

layout (push_constant) uniform PushConstants {
    mat4 model;
} pcs;

void main() {
    fragColor = inColor;
    texCoord = inCoord;
    gl_Position = uniforms.proj * uniforms.view * pcs.model * vec4(inPosition, 1);
}