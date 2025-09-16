#version 460

layout(location = 0) in  vec2 vNdc;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneData {
    mat4 view; // view transforms world -> view
    mat4 proj; // proj transforms view -> clip
    mat4 viewproj; // (unused)
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
} scene;

layout(set = 1, binding = 0) uniform samplerCube uEnv;

vec3 reconstructViewDir(vec2 ndc) {
    // Clip space pos on far plane (z=1, w=1)
    vec4 clip = vec4(ndc, 1.0, 1.0);

    // Inverse projection to view space
    mat4 invProj = inverse(scene.proj);
    vec3 v = (invProj * clip).xyz;

    // For a perspective projection, the direction is from the origin through this point
    return normalize(v);
}

void main() {
    // Remove translation from the view matrix; only rotation matters for directions.
    mat3 R = mat3(scene.view);
    mat3 invR = transpose(R); // inverse of rotation

    vec3 dirView  = reconstructViewDir(vNdc);
    vec3 dirWorld = invR * dirView;

    vec3 env = texture(uEnv, dirWorld).rgb;

    // Simple Reinhard tone map
    // This looks very wrong, maybe using sRGB swapchain also makes this redundant, unsure
    env = env / (env + vec3(1.0));

    outColor = vec4(env, 1.0);
}
