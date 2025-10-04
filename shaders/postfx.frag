#version 450

layout(set = 0, binding = 0) uniform sampler2D uHDR; // R16G16B16A16_SFLOAT, linear

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    outColor = texelFetch(uHDR, px, 0);
}
