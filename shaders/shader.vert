#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model; mat4 view; mat4 proj;
} uniforms;

void main() {
    fragColor = inColor;
    gl_Position =uniforms.proj*uniforms.view*uniforms.model * vec4(inPosition, 0, 1) * vec4(1,1,1,1);
}