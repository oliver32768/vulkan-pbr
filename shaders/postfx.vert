#version 450

// call vkCmdDraw(3, 1, 0, 0)

layout(location = 0) out vec2 vUV;

void main() {
    // 3 clip-space verts: (-1,-1), (3,-1), (-1,3)
    const vec2 pos[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 p = pos[gl_VertexIndex];
    gl_Position = vec4(p, 0.0, 1.0);

    // Map clip-space [-1,1] to UV [0,1]
    vUV = p * 0.5 + 0.5;
}
