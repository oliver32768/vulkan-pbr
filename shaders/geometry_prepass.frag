#version 450

layout(set = 1, binding = 0) uniform GLTFMaterialData {
    vec4 colorFactors;
    vec4 metal_rough_factors;
    vec4 emissiveFactors;
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;
layout(set = 1, binding = 3) uniform sampler2D normalTex; // optional, see note below
layout(set = 1, binding = 4) uniform sampler2D AOTex;
layout(set = 1, binding = 5) uniform sampler2D emissiveTex;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorldNormal;
layout(location = 3) in vec4 vVertexColor;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

vec3 safeNormalize(vec3 v) { return normalize(v + 1e-8); }

mat3 buildTBN(vec3 N, vec3 pos, vec2 uv) {
    vec3 dp1 = dFdx(pos);
    vec3 dp2 = dFdy(pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 T = normalize(duv2.y * dp1 - duv1.y * dp2);
    vec3 B = normalize(-duv2.x * dp1 + duv1.x * dp2);

    // Orthonormalize
    T = normalize(T - N * dot(N, T));
    B = cross(N, T);
    return mat3(T, B, N);
}

void main() {
    // Base color / albedo (modulate by vertex color)
    vec4 baseColorTex = texture(colorTex, vUV);
    vec4 baseColor = baseColorTex * materialData.colorFactors * vVertexColor;
    if (baseColor.w < 0.25) {
        discard;
    }
    outAlbedo = baseColor; // includes alpha

    // Normals
    vec3 N = safeNormalize(vWorldNormal);
    mat3 TBN = buildTBN(N, vWorldPos, vUV); // tangent to world
    vec3 nm = texture(normalTex, vUV).xyz * 2.0 - 1.0; // assume linear texture, default (0.5, 0.5, 1)
    N = normalize(TBN * nm);
    outNormal = vec4(N, 1.0); // world space

    // Metallic/Roughness/AO/Emissive packing:
    //   .r = metallic
    //   .g = roughness
    //   .b = ao
    //   .a = emissiveIntensity (or unused)
    vec2 mrTex = texture(metalRoughTex, vUV).gb; // roughness.g, metallic.b
    float metallic = mrTex.y * materialData.metal_rough_factors.x;
    float roughness = mrTex.x * materialData.metal_rough_factors.y;

    float ao = texture(AOTex, vUV).r;
    vec3 emissive = texture(emissiveTex, vUV).rgb * materialData.emissiveFactors.rgb;
    float emissiveI = max(max(emissive.r, emissive.g), emissive.b); // intensity proxy

    outMaterial = vec4(metallic, roughness, ao, emissiveI);
}
