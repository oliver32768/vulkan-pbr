#version 460 core

layout (location = 0) in vec3 vWorldPos;
layout (location = 1) in vec3 vWorldNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vVertexColor;

layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; 
	vec4 sunlightColor;
} sceneData;

layout(set = 1, binding = 0) uniform sampler2D colorTex;

void main() {             
    vec4 baseSample = texture(colorTex, vUV);
    if (baseSample.w < 0.25) {
        discard;
    }
}  