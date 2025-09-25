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

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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

// Returns an exact shadow bias in the space of whatever texelWidth is in.
// N and L are expected to be normalized
float GetShadowBias(vec3 N, vec3 L, float texelWidth) {
  const float sqrt2 = 1.41421356; // Mul by sqrt2 to get diagonal length
  const float quantize = 2.0 / (1 << 23); // Arbitrary constant that should help prevent most numerical issues
  const float b = sqrt2 * texelWidth / 2.0;
  const float NoL = clamp(abs(dot(N, L)), 0.0001, 1.0);
  return quantize + b * length(cross(N, L)) / NoL;
}

int selectCascade(float z) {
    // csm.splitDepths = [znear, s1, ..., zfar] in the SAME space as vViewDepth
    for (int i = 0; i < NUM_CASCADES; ++i) {
        if (z < csm.splitDepths[i+1]) {
            return i;
        }
    }
    return NUM_CASCADES - 1;
}

const vec2 kPoisson16[16] = vec2[](
  vec2( 0.2853, -0.0588), vec2(-0.1888,  0.2231),
  vec2(-0.0421, -0.3219), vec2( 0.3458,  0.2309),
  vec2(-0.3010, -0.1195), vec2( 0.1063,  0.3539),
  vec2(-0.3567,  0.0847), vec2( 0.0087,  0.1183),
  vec2( 0.2032, -0.3193), vec2(-0.0796,  0.3586),
  vec2(-0.2643, -0.3026), vec2( 0.3536, -0.1902),
  vec2(-0.0063, -0.0318), vec2( 0.1181,  0.1085),
  vec2(-0.1979, -0.0089), vec2( 0.2670,  0.0174)
);

// Cheap per-pixel hash -> [0,1)
float hash12(vec2 p) {
  // keep it branchless and stable across frames
  vec3 p3  = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

mat2 rot2(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, -s, s, c);
}

float ShadowCalculation(vec3 fragPosWorldSpace, vec3 normal, vec3 lightDir) {
    vec4 fragPosViewSpace = sceneData.view * vec4(fragPosWorldSpace, 1.0);
    float depthValue = -fragPosViewSpace.z;
    int layer = selectCascade(depthValue);
    vec4 fragPosLightSpace = csm.lightViewProj[layer] * vec4(fragPosWorldSpace, 1.0);

    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    vec2 uv = projCoords.xy * 0.5 + 0.5; // map from [-1,1] to [0,1]
    float currentDepth = projCoords.z; // Vulkan NDC z is already [0,1]; don't remap

    if (currentDepth > 1.0) return 0.0;

    const float pcfRadiusTexels = 2.5;
    vec2 shadowRes = vec2(textureSize(uShadowMap, 0).xy);
    vec2 worldPerTex = csm.orthoXY[layer] / shadowRes;
    float texelWidthWorld = worldPerTex.x;
    float texelWidthForBias = texelWidthWorld * pcfRadiusTexels;

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float texelWidth = texelSize.x;

    float bias = GetShadowBias(normalize(normal), normalize(lightDir), texelWidth);

    float angle = 6.2831853 * hash12(gl_FragCoord.xy);
    mat2 R = rot2(angle);
    float jitter = mix(0.9, 1.1, hash12(gl_FragCoord.yx * 1.37));
    float sum = 0.0;
    float wsum = 0.0;
    vec2 radius = texelSize * (pcfRadiusTexels * jitter);

    for (int i = 0; i < 16; ++i) {
        vec2 o = R * kPoisson16[i]; // rotate pattern
        vec2 duv = o * radius; // scale to desired footprint

        float pcfDepth = texture(uShadowMap, vec3(uv + duv, layer)).r;

        // tent weight (closer samples contribute more)
        float r = length(o); // in [0,1] roughly (unit disk)
        float w = 1.0 - r; // simple tent; can use smoothstep
        wsum += w;

        sum += ((currentDepth + bias) < pcfDepth ? 1.0 : 0.0) * w;
    }

    float shadow = sum / max(wsum, 1e-5);
    return shadow;
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// Point light shaded with the same GGX/Smith BRDF used for the sun
vec3 shadePointLight(Light light, vec3 P, vec3 N, vec3 V, float NoV, vec3 baseColor, vec3 F0, float alpha, float metallic){
    // Vector from fragment to light
    vec3 L = light.pos_radius.xyz - P;
    float d2 = max(dot(L, L), 1e-8);
    float d  = sqrt(d2);
    L /= d;

    float NoL = clamp(dot(N, L), 0.0, 1.0);
    if (NoL <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    // Microfacet terms
    float D = D_GGX(NoH, alpha);
    float G = V_SmithGGXCorrelated(NoV, NoL, alpha);
    vec3  F = F_Schlick(VoH, F0);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 Fr = D * G * F;
    vec3 Fd = kD * (baseColor / PI);

    vec3 Li = light.color_intensity.rgb * light.color_intensity.a;

    float attenuation = 1.0 / d2;

    // smooth range falloff using pos_radius.w as range
    float range = light.pos_radius.w;
    if (range > 0.0) {
        // Smoothly fade to zero near the edge; keeps inverse-square behavior at the core.
        float x = clamp(d / range, 0.0, 1.0);
        float s = (1.0 - x*x*x*x); // (1 - (d/R)^4)
        attenuation *= s * s; // square for a softer knee
    }

    return (Fd + Fr) * Li * NoL * attenuation;
}

uint getDepthSlice(float zAbs, uint numSlices, float zNear, float zFar) {
    float nz = max(zAbs, 1e-6);
    float l  = log(zFar / zNear);
    float t  = (log(nz) - log(zNear)) / l;   // [0,1]
    uint s   = uint(floor(t * float(numSlices)));
    return clamp(s, 0u, numSlices - 1u);
}

uint clusterIndexFromWorldPos(vec3 worldPos, float zNear, float zFar) {
    uvec2 px = uvec2(gl_FragCoord.xy - vec2(0.5));
    uint tileSize = u.gridDim.w;
    uint tileX = min(px.x / tileSize, u.gridDim.x - 1u);
    uint tileY = min(px.y / tileSize, u.gridDim.y - 1u);

    // View-space Z (choose the right source)
    vec3 viewPos = vec3(sceneData.view * vec4(worldPos, 1.0));

    uint z = getDepthSlice(abs(viewPos.z), u.gridDim.z, zNear, zFar);

    return (z * u.gridDim.y + tileY) * u.gridDim.x + tileX;
}

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
    const float MAX_REFLECTION_LOD = floor(log2(1024)) + 1; // use `textureQueryLevels`?
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
    //outFragColor = vec4(vec3(count / 128.0), alphaOut);
    //outFragColor = vec4(vec3(clusterIndex / (16.0 * 9.0 * 24.0)), alphaOut);
    //outFragColor = vec4(materialData.emissiveFactors.xyz, alphaOut);
}
