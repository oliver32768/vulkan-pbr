#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures_pbr.glsl"

layout (location = 0) in vec3 vWorldPos;
layout (location = 1) in vec3 vWorldNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vVertexColor;

layout (location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// NDF: GGX (Trowbridge-Reitz), Isotropic
// G(h) = a^2 / (pi * ((n.h)^2 * (a^2 - 1) + 1)^2
float distributionGGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometrySchlickGGX(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometrySmith(float NdotV, float NdotL, float k) {
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

// build TBN from screen-space derivatives (no tangents in mesh needed)
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
    // camera position from inverse(view)
    vec3 camPos = inverse(sceneData.view)[3].xyz;

    // base inputs
    vec4 baseSample = texture(colorTex, vUV); // todo: srgb
    vec3 baseColor = baseSample.rgb * materialData.colorFactors.rgb * vVertexColor;
    float alphaOut = baseSample.a * materialData.colorFactors.a;

    // metallic & roughness. Roughness = G, Metalness = B
    vec2 mrTex = texture(metalRoughTex, vUV).gb; 
    float roughness = clamp(mrTex.x * materialData.metal_rough_factors.y, 0.04, 1.0);
    float metallic = clamp(mrTex.y * materialData.metal_rough_factors.x, 0.0, 1.0);
    float alpha = roughness * roughness; // perceptual to microfacet alpha

    // normal mapping
    vec3 N = normalize(vWorldNormal);
    vec3 nm = texture(normalTex, vUV).xyz * 2.0 - 1.0; // assume linear texture, default (0.5,0.5,1)
    mat3 TBN = buildTBN(N, vWorldPos, vUV);
    N = normalize(TBN * nm);

    // AO
    float ao = texture(AOTex, vUV).r; // default white => 1.0

    // emissive
    vec3 emissive = texture(emissiveTex, vUV).rgb; // expect sRGB view
    // todo: multiply by emissive factor, strength, etc.

    // lighting vectors (world space)
    vec3 V = normalize(camPos - vWorldPos);
    vec3 L = normalize(sceneData.sunlightDirection.xyz); // assume this is the *incoming* dir
    vec3 H = normalize(V + L);

    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);

    // Fresnel reflectance at normal incidence
    vec3  F0 = mix(vec3(0.04), baseColor, metallic);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float D = distributionGGX(NdotH, alpha);
    float k = (roughness + 1.0);
    k = (k * k) / 8.0; // Schlick-GGX remap
    float G = geometrySmith(NdotV, NdotL, k);

    vec3 specular = (D * G) * F / max(4.0 * NdotV * NdotL, 1e-4);

    // Lambertian diffuse (metallic scales it down)
    vec3 kd = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kd * baseColor / PI;

    // directional light color & intensity
    vec3  sunColor = sceneData.sunlightColor.rgb;
    float sunI = sceneData.sunlightColor.w; // intensity in .w 

    vec3 direct = (diffuse + specular) * sunColor * sunI * NdotL;

    // ambient (todo: IBL)
    vec3 ambient = sceneData.ambientColor.rgb * baseColor * ao;

    vec3 color = direct + ambient + emissive;

    outFragColor = vec4(color, alphaOut);
}
