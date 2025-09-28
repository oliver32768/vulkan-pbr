#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

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
	vec4 emissiveFactors;
} materialData;
layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
layout(set = 1, binding = 3) uniform sampler2D normalTex;
layout(set = 1, binding = 4) uniform sampler2D AOTex;
layout(set = 1, binding = 5) uniform sampler2D emissiveTex;

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
    Vertex vertices[];
};

layout(push_constant) uniform constants {
    mat4 render_matrix; // model matrix
    VertexBuffer vertexBuffer;
} PushConstants;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorldNormal;
layout(location = 3) out vec4 vVertexColor;

void main() {
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

    vec4 worldPos = PushConstants.render_matrix * vec4(v.position, 1.0);
    vec3 worldN = normalize(mat3(PushConstants.render_matrix) * v.normal);

    vWorldPos = worldPos.xyz;
    vUV = vec2(v.uv_x, v.uv_y);
    vWorldNormal = worldN;
    vVertexColor = v.color;

    gl_Position = sceneData.viewproj * worldPos;
}