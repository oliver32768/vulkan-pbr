layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; // w for sun power
	vec4 sunlightColor;
} sceneData;

layout(set = 1, binding = 0) uniform samplerCube uEnv;
layout(set = 1, binding = 1) uniform samplerCube uIrradiance;
layout(set = 1, binding = 2) uniform samplerCube uPrefiltered;
layout(set = 1, binding = 3) uniform sampler2D uBrdfLUT;

layout(set = 2, binding = 0) uniform sampler2DArray uShadowMap;

#define NUM_CASCADES 5

layout(set = 3, binding = 0) uniform Cascades {
	mat4 lightViewProj[NUM_CASCADES]; // world -> light clip per cascade
	float splitDepths[NUM_CASCADES + 1]; // view-space split planes (near..far)
	vec2 orthoXY[NUM_CASCADES];
} csm;

struct Light {
	vec4 pos_radius; // xyz = view/world position, w = radius
	vec4 color_intensity; // rgb = color, w = intensity
};

layout(std430, set = 4, binding = 0) readonly buffer LightsBuf {
	Light lights[]; // all point lights in the scene
};

layout(set = 4, binding = 1) uniform Params {
	uvec4 lightCount; // numLights(.x)
	uvec4 gridDim; // numTiles(.xyz), tileSizePx(.w)
	uvec4 screenDim; // (.xy)
	vec4 frustumPlanes; // zNear(.x), zFar(.y)
} u;

struct LightGrid {
	uint offset;
	uint count;
};

layout(std430, set = 4, binding = 2) readonly buffer LightGridSSBO {
	LightGrid lightGrid[];
};
layout(std430, set = 4, binding = 3) readonly buffer LightIndexSSBO {
	uint globalLightIndexList[];
};

layout(set = 5, binding = 0) uniform sampler2D gAlbedo; // RGB = baseColor, A optional
layout(set = 5, binding = 1) uniform sampler2D gNormal; // XYZ = world normal in [-1,1] encoded to [0,1]
layout(set = 5, binding = 2) uniform sampler2D gMaterial; // R=metal, G=rough, B=AO
layout(set = 5, binding = 3) uniform sampler2D gDepth; // view-space depth (positive distance), single channel
layout(set = 5, binding = 4) uniform sampler2D gGeoNormal; // geo normal
layout(set = 5, binding = 5) uniform sampler2D gEmissive; // emissive

layout(push_constant) uniform constants {
	ivec4 screenSize;
} pc;