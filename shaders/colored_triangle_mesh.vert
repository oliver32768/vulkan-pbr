#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 

// buffer_reference = used from buffer address
layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants {	
	mat4 render_matrix;
	VertexBuffer vertexBuffer; // stored as uint64 due to buffer_reference layout specifier on VertexBuffer declaration
} PushConstants;

void main() {	
	//load vertex data from device adress
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex]; // second "." operator is dereferencing pointer here

	//output data
	gl_Position = PushConstants.render_matrix * vec4(v.position, 1.0f); // model to world
	outColor = v.color.xyz;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
}
