#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures_pbr.glsl"

layout (location = 0) in vec3 vWorldPos;
layout (location = 1) in vec3 vWorldNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vVertexColor;

layout (location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;
#define MEDIUMP_FLT_MAX 65504.0
#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)

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

// F
vec3 F_Schlick(float u, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - u, 5.0);
}

// D
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * f * f);
}

// G
float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
    float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    return 0.5 / (GGXV + GGXL);
}

vec3 reinhardTonemap(vec3 color) {
    return color / (color + vec3(1.0));
}

void main() {
    vec3 camPos = inverse(sceneData.view)[3].xyz;

    vec4 baseSample = texture(colorTex, vUV);
    vec3 baseColor = baseSample.rgb * materialData.colorFactors.rgb * vVertexColor;
    float alphaOut = baseSample.a * materialData.colorFactors.a;

    // Roughness = G, Metalness = B
    vec2 mrTex = texture(metalRoughTex, vUV).gb; 
    float roughness = clamp(mrTex.x * materialData.metal_rough_factors.y, 0.04, 1.0);
    float metallic = clamp(mrTex.y * materialData.metal_rough_factors.x, 0.0, 1.0);

    vec3 N = normalize(vWorldNormal);
    vec3 nm = texture(normalTex, vUV).xyz * 2.0 - 1.0; // assume linear texture, default (0.5,0.5,1)
    mat3 TBN = buildTBN(N, vWorldPos, vUV);
    N = normalize(TBN * nm);

    float ao = texture(AOTex, vUV).r;

    vec3 emissive = texture(emissiveTex, vUV).rgb; 

    vec3 V = normalize(camPos - vWorldPos); // towards camera
    vec3 L = normalize(sceneData.sunlightDirection.xyz); 
    vec3 H = normalize(V + L);

    float NoV = abs(dot(N, V)) + 1e-5;
    float NoL = clamp(dot(N, L), 0.0, 1.0); // cos(theta_i)
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float LoH = clamp(dot(L, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    // perceptually linear roughness to roughness (see parameterization)
    float alpha = roughness * roughness;

    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    // Specular
    float D = D_GGX(NoH, alpha);
    vec3 F = F_Schlick(VoH, F0);
    float G = V_SmithGGXCorrelated(NoV, NoL, alpha);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 Fr = (D * G * F); // 1.0 / max(4.0 * NoV * NoL, 1e-4) is already part of the G we calculate
    vec3 Fd = kD * baseColor / PI;

    // directional light color & intensity
    vec3 L_i = sceneData.sunlightColor.rgb * sceneData.sunlightColor.w;
    float cos_i = NoL;

    // TODO: I think this linear separation is what devsh says is wrong
    // f_r = F(H,L) * G(H,L,V) * D(H) / (4 * dot(V,N) * dot(L,N)) + (1 - F(N,L)) * (1 - F(N,V)) * (baseColor / PI)
    vec3 direct = (Fd + Fr) * L_i * cos_i;

    //vec3 ambient = sceneData.ambientColor.rgb * baseColor * ao;

    vec3 color = direct + emissive;

    outFragColor = vec4(color, alphaOut);
}
