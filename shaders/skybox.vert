#version 460

layout(location = 0) out vec2 vNdc;  // pass NDC xy to the fragment

// No vertex inputs needed.

void main() {
    // Full-screen triangle in clip space: (-1,-1), (3,-1), (-1,3)
    // gl_VertexIndex: 0,1,2
    const vec2 verts[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 pos = verts[gl_VertexIndex];
    vNdc = pos; // NDC xy == clip.xy since w=1
    gl_Position = vec4(pos, 1.0, 1.0); // z = w -> depth = 1.0 (far)
}
