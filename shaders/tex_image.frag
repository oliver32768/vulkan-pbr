//glsl version 4.5
#version 450

//shader input
layout (location = 0) in vec3 inColor; // this is just for compatibility with the outputs of the old vertex shader (i.e. model-space normals as color)
layout (location = 1) in vec2 inUV;
//output write
layout (location = 0) out vec4 outFragColor;

//texture to access
layout(set = 0, binding = 0) uniform sampler2D displayTexture;

void main() 
{
	outFragColor = texture(displayTexture, inUV);
}
