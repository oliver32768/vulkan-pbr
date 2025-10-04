#version 450

layout(set = 0, binding = 0) uniform sampler2D uHDR; // R16G16B16A16_SFLOAT, scene-linear

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// ACES
const mat3 ACESInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

const mat3 ACESOutputMat = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (v * 0.983729 + 0.4329510) + 0.238081;
    return a / b;
}

vec3 TonemapACES(vec3 c) {
    c = ACESInputMat * c;
    c = RRTAndODTFit(c);
    c = ACESOutputMat * c;
    return clamp(c, 0.0, 1.0);
}

// Reinhard
vec3 TonemapReinhard(vec3 c) {
    return c / (1.0 + c);
}

// Uchimura / Gran Turismo
// impl. src: https://gist.github.com/Pikachuxxxx/136940d6d0d64074aba51246f514bd26
vec3 gt_uchimura(vec3 x, float P, float a, float m, float l, float c, float b) {
    float l0 = ((P - m) * l) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    vec3 w0 = vec3(1.0 - smoothstep(0.0, m, x)); // toe region
    vec3 w2 = vec3(step(m + l0, x)); // shoulder
    vec3 w1 = vec3(1.0) - w0 - w2; // linear mid

    vec3 T = m * pow(x / m, vec3(c)) + b; // toe (power)
    vec3 L = m + a * (x - m); // linear mid
    vec3 S = vec3(P - (P - S1) * exp(CP * (x - S0))); // shoulder (exp)

    return clamp(T * w0 + L * w1 + S * w2, 0.0, 1.0);
}

vec3 TonemapUchimura(vec3 x) {
    const float P = 1.0; // display peak in normalized units (keep 1.0 for SDR)
    const float a = 1.0; // mid slope (contrast)
    const float m = 0.22; // start of linear section
    const float l = 0.40; // length of linear section
    const float c = 1.33; // toe strength
    const float b = 0.0; // toe offset (pedestal)
    return gt_uchimura(x, P, a, m, l, c, b);
}

// Hable (Uncharted2 filmic)
vec3 TonemapHable(vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    const float W = 11.2; 

    vec3 color = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
    color /= white;
    return color;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 hdr = texelFetch(uHDR, px, 0);

    // Exposure in linear (pc.exposure ~ pow(2.0, EV))
    float exposure = 0.5;
    vec3 c = hdr.rgb * exposure;

    // Tonemap
    //c = TonemapReinhard(c);
    //c = TonemapHable(c);
    //c = TonemapACES(c);
    c = TonemapUchimura(c);

    // sRGB swapchain img will encode on write
    outColor = vec4(c, hdr.a);
}
