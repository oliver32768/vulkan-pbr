#version 460

layout(location = 0) out vec2 vUV;

void main() {
    // Full-screen triangle (-1,-1), (3,-1), (-1,3)
    const vec2 verts[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 pos = verts[gl_VertexIndex];
    vec2 uv = pos * 0.5 + 0.5;
    vUV = uv;

    gl_Position = vec4(pos, 0.0, 1.0); // z doesnt matter
}
