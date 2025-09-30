#version 460

#extension GL_GOOGLE_include_directive : require
#include "input_structures_pbr.glsl"

layout (location = 0) in vec3 vWorldPos;
layout (location = 1) in vec3 vWorldNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vVertexColor;

layout (location = 0) out vec4 outFragColor;

#include "fragment_pbr.glsl"

void main() {
    vec3 camPos = inverse(sceneData.view)[3].xyz;

    vec4 baseSample = texture(colorTex, vUV);

    if (baseSample.w < 0.25) {
        discard;
    }

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

    vec3 emissive = materialData.emissiveFactors.xyz * texture(emissiveTex, vUV).rgb; 

    vec3 V = normalize(camPos - vWorldPos); // towards camera
    vec3 L = normalize(sceneData.sunlightDirection.xyz); 
    vec3 H = normalize(V + L);
    vec3 R = reflect(-V, N);   

    float NoV = abs(dot(N, V)) + 1e-5;
    float NoL = clamp(dot(N, L), 0.0, 1.0); // cos(theta_i)
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float LoH = clamp(dot(L, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    // perceptually linear roughness to roughness
    float alpha = roughness * roughness;

    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    // Specular
    float D = D_GGX(NoH, alpha);
    vec3 F = F_Schlick(VoH, F0);
    float G = V_SmithGGXCorrelated(NoV, NoL, alpha);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic); 
    //vec3 kD = (vec3(1.0) - F_Schlick(NoL, F0)) * (vec3(1.0) - F_Schlick(NoV, F0)); // devsh: "(1 - Fresnel(N,L)) * (1 - Fresnel(N,V))"
    vec3 Fr = (D * G * F); // 1.0 / max(4.0 * NoV * NoL, 1e-4) is already part of the G we calculate
    vec3 Fd = kD * (baseColor / PI);

    // --- Directional light ---
    vec3 L_i = sceneData.sunlightColor.rgb * sceneData.sunlightColor.w;
    float cos_i = NoL;

    vec3 direct = (Fd + Fr) * L_i * cos_i;

    // --- IBL ---
    vec3 F_NV = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kS_ibl = F_NV;
    vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic); // not devsh
    vec3 irradiance = texture(uIrradiance, N).rgb;
    vec3 diffuse = irradiance * (baseColor / PI);
    int maxMip = textureQueryLevels(uPrefiltered) - 1;
    vec3 prefilteredColor = textureLod(uPrefiltered, R, roughness * maxMip).rgb; // idk if this should be roughness or roughness^2
    vec2 envBRDF = texture(uBrdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg; // same here
    vec3 specular = prefilteredColor * (F_NV * envBRDF.x + envBRDF.y);
    vec3 ambient = (kD_ibl * diffuse + specular) * ao; 

    // --- Shadow Mapping ---
    float shadow = ShadowCalculation(vWorldPos, normalize(vWorldNormal), L); // use geometric normal 
    vec3 color = emissive + ambient + ((1.0 - shadow) * direct);

    // --- Point lights ---
    uint clusterIndex = clusterIndexFromWorldPos(vWorldPos, u.frustumPlanes.x, u.frustumPlanes.y);

    LightGrid lg = lightGrid[clusterIndex];
    uint base = lg.offset;
    uint count = lg.count;

    for (uint i = 0u; i < count; ++i) {
        uint li = globalLightIndexList[base + i];
        Light L = lights[li];
        color += shadePointLight(L, vWorldPos, N, V, NoV, baseColor, F0, alpha, metallic);
    }

    outFragColor = vec4(color, alphaOut);
    //outFragColor = vec4(metallic, roughness, ao, 1.0);
}
