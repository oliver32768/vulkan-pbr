layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; // w for sun power
	vec4 sunlightColor;
} sceneData;

layout(set = 1, binding = 0) uniform GLTFMaterialData {
	vec4 colorFactors;
	vec4 metal_rough_factors;
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
layout(set = 1, binding = 3) uniform sampler2D normalTex;
layout(set = 1, binding = 4) uniform sampler2D AOTex;
layout(set = 1, binding = 5) uniform sampler2D emissiveTex;

layout(set = 2, binding = 0) uniform samplerCube uEnv;
layout(set = 2, binding = 1) uniform samplerCube uIrradiance;
layout(set = 2, binding = 2) uniform samplerCube uPrefiltered;
layout(set = 2, binding = 3) uniform sampler2D uBrdfLUT;

layout(set = 3, binding = 0) uniform sampler2DArray uShadowMap;

#define NUM_CASCADES 5

layout(set = 4, binding = 0) uniform Cascades {
	mat4 lightViewProj[NUM_CASCADES]; // world -> light clip per cascade
	float splitDepths[NUM_CASCADES + 1]; // view-space split planes (near..far)
	vec2 orthoXY[NUM_CASCADES];
} csm;

struct Light {
	vec4 pos_radius; // xyz = view/world position, w = radius
	vec4 color_intensity; // rgb = color, w = intensity
	// add flags/type/shadow idx in another uint
};

layout(std430, set = 5, binding = 0) readonly buffer LightsBuf {
	Light lights[]; // all point lights in the scene
};
layout(std430, set = 5, binding = 1) readonly buffer ClusterOffsets {
	uint offsets[]; // size = numClusters + 1
};
layout(std430, set = 5, binding = 2) readonly buffer ClusterIndices {
	uint indices[]; // concatenated light indices (CSR)
};
layout(set = 5, binding = 3) uniform Params {
	uint lightCount; // used when clustering is OFF
	uvec3 gridDim; // when clustering is ON
	uint useClusters; // 0/1
} u;