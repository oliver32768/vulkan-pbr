#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures_deferred.glsl"

layout(location = 0) in vec2 vUV;
layout (location = 0) out vec4 outFragColor;

#include "fragment_pbr.glsl"

vec3 worldDirFromNDC(vec2 ndc, mat4 invProj, mat4 invView) {
    // point on far plane in clip (z=1) -> view, then make it a direction
    vec4 v = invProj * vec4(ndc, 1.0, 1.0);
    vec3 dirView = normalize(v.xyz / v.w);

    // transform direction to world; ignore translation by using w=0
    vec3 dirWorld = normalize((invView * vec4(dirView, 0.0)).xyz);
    return dirWorld;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy); // pixel coords in the big (monitor-sized) RTs

    float depth = texelFetch(gDepth, px, 0).r; // hardware depth from the depth attachment

    mat4 invViewProj = inverse(sceneData.viewproj);
    mat4 invView = inverse(sceneData.view);
    mat4 invProj = inverse(sceneData.proj);

    const float eps = 1e-6;
    bool isBackground = (depth <= 0.0 + eps);

    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / vec2(pc.screenSize.xy);
    vec2 ndcXY = uv * 2.0 - 1.0;

    if (isBackground) {
        vec3 dirWorld = worldDirFromNDC(ndcXY, invProj, invView);
        vec3 env = textureLod(uEnv, dirWorld, 0.0).rgb;
        outFragColor = vec4(env, 1.0);
        return;
    }

    // fetch G-buffers
    vec4  albedoSample = texelFetch(gAlbedo, px, 0);

    // normals are stored as world-space float, no remap:
    vec3 vWorldNormal = normalize(texelFetch(gNormal, px, 0).xyz);

    // materials already include factors:
    vec4 M = texelFetch(gMaterial, px, 0);
    float metallic = clamp(M.r, 0.0, 1.0);
    float roughness = clamp(M.g, 0.04, 1.0);
    float ao = clamp(M.b, 0.0, 1.0);
    float emissiveI = max(0.0, M.a);

    // baseColor already has colorFactors * vertexColor applied in G-buffer:
    vec3 baseColor  = albedoSample.rgb;
    float alphaOut = albedoSample.a;

    // reconstruct world pos (atp should I just make a pos gbuffer idk)
    vec4 clipOrNdc = vec4(ndcXY, depth, 1.0); // NDC coords
    vec4 worldH = invViewProj * clipOrNdc;
    vec3 vWorldPos = worldH.xyz / worldH.w;
    vec3 camPos = invView[3].xyz;

    // emissive is intensity only
    vec3 emissive = vec3(emissiveI);

    // Lighting
    vec3 V = normalize(camPos - vWorldPos);
    vec3 R = reflect(-V, vWorldNormal);
    vec3 L = normalize(sceneData.sunlightDirection.xyz); 
    vec3 H = normalize(V + L); 

    float NoV = abs(dot(vWorldNormal, V)) + 1e-5;
    float NoL = clamp(dot(vWorldNormal, L), 0.0, 1.0);
    float NoH = clamp(dot(vWorldNormal, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    float alpha = roughness * roughness;
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float D = D_GGX(NoH, alpha);
    vec3  F = F_Schlick(VoH, F0);
    float G = V_SmithGGXCorrelated(NoV, NoL, alpha);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic); 
    vec3 Fr = (D * G * F);
    vec3 Fd = kD * (baseColor / PI);

    // Directional light
    vec3 L_i = sceneData.sunlightColor.rgb * sceneData.sunlightColor.w;
    vec3 direct = (Fd + Fr) * L_i * NoL;

    // IBL
    vec3 F_NV = fresnelSchlickRoughness(max(dot(vWorldNormal, V), 0.0), F0, roughness);
    vec3 kS_ibl = F_NV;
    vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
    vec3 irradiance = texture(uIrradiance, vWorldNormal).rgb;
    vec3 diffuse = irradiance * (baseColor / PI);
    int  maxMip = textureQueryLevels(uPrefiltered) - 1;
    vec3 prefilteredColor = textureLod(uPrefiltered, R, roughness * maxMip).rgb;
    vec2 envBRDF = texture(uBrdfLUT, vec2(max(dot(vWorldNormal, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F_NV * envBRDF.x + envBRDF.y);
    vec3 ambient = (kD_ibl * diffuse + specular) * ao; 

    // Shadow Mapping
    float shadow = ShadowCalculation(vWorldPos, normalize(vWorldNormal), L);
    vec3 color = emissive + ambient + ((1.0 - shadow) * direct);

    // Point lights (clustered)
    uint clusterIndex = clusterIndexFromWorldPos(vWorldPos, u.frustumPlanes.x, u.frustumPlanes.y);
    LightGrid lg = lightGrid[clusterIndex];
    uint base = lg.offset;
    uint count = lg.count;
    for (uint i = 0u; i < count; ++i) {
        uint li = globalLightIndexList[base + i];
        Light Lpt = lights[li];
        color += shadePointLight(Lpt, vWorldPos, vWorldNormal, V, NoV, baseColor, F0, alpha, metallic);
    }

    outFragColor = vec4(color, alphaOut);
}