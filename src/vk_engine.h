// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <camera.h>

struct EngineStats {
	float frametime;
	int triangle_count;
	int drawcall_count;
	float scene_update_time;
	float mesh_draw_time;
};

struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;
	VkDescriptorSetLayout materialLayout;

	MaterialPipeline zPrepassPipeline;
	VkDescriptorSetLayout zPrepassLayout;

	struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
		// "In vulkan, when you want to bind a uniform buffer, it needs to meet a minimum requirement for its alignment. 
		// 256 bytes is a good default alignment for this which all the gpus we target meet, so we are adding those vec4s to pad the structure to 256 bytes."
		glm::vec4 extra[14]; 
	};

	// will be written/bound to descriptor
	struct MaterialResources {
		AllocatedImage colorImage;
		VkSampler colorSampler;

		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;

		AllocatedImage normalImage;
		VkSampler normalSampler;

		AllocatedImage occlusionImage;
		VkSampler occlusionSampler;

		AllocatedImage emissiveImage;
		VkSampler emissiveSampler;

		VkBuffer dataBuffer; // material constants 
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_z_prepass_pipeline(VulkanEngine* engine);

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_z_prepass_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node {
	std::shared_ptr<MeshAsset> mesh;
	// TODO: again, it shouldn't be hard to get rid of runtime polymorphism here
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct RenderObject {
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* material;
	MaterialInstance* zPrepassMaterial;
	
	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;

	Bounds bounds;
};

// render 'passes' (not the vulkan ones)
// change drawgeometry respectively
struct DrawContext {
	std::vector<RenderObject> OpaqueSurfaces;
	std::vector<RenderObject> TransparentSurfaces;
};

struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct LightInfo {
	glm::mat4 lightViewProj;
	glm::vec2 orthoDims;
};

struct GPUSceneData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; // w for sun power
	glm::vec4 sunlightColor;
};

constexpr uint32_t NUM_CASCADES = 5;

struct alignas(16) Std140Float {
	float v;
	float _pad[3]; // 16-byte stride
};
static_assert(sizeof(Std140Float) == 16);

struct alignas(16) Std140Vec2 {
	glm::vec2 v;
	float _pad[2]; // 16-byte stride
};
static_assert(sizeof(Std140Vec2) == 16);

struct alignas(16) GPUShadowCascades {
	glm::mat4 lightViewProj[NUM_CASCADES]; // 64 each
	Std140Float splitDepths[NUM_CASCADES + 1]; // 16 stride
	Std140Vec2 orthoDims[NUM_CASCADES]; // 16 stride
};

static_assert(sizeof(glm::mat4) == 64);
static_assert(sizeof(GPUShadowCascades) == 64 * NUM_CASCADES + 16 * (NUM_CASCADES + 1) + 16 * NUM_CASCADES); // 496 when N=5

struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct DeletionQueue {
	// TODO: if you need to delete thousands of objects and want them deleted faster, 
	// a better implementation would be to store arrays of vulkan handles of various types such as VkImage, VkBuffer, and so on. 
	// And then delete those from a loop.
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); // call functors
		}

		deletors.clear();
	}
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct FrameData {
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkSemaphore _swapchainSemaphore; // swapchain image acquisition from OS
	VkFence _renderFence; // lets us wait for the draw commands of a given frame to be finished

	DeletionQueue _deletionQueue; 
	DescriptorAllocatorGrowable _frameDescriptors;
};

struct IrradiancePushConstants {
	uint32_t size;
	uint32_t sampleCount;
};

struct PrefilterPushConstants {
	uint32_t size;
	uint32_t sampleCount;
	uint32_t mipLevel;
	float roughness;
};

struct IBLResources {
	DescriptorAllocatorGrowable descriptorAllocator;

	AllocatedImage equirect; // 2D HDR input
	AllocatedImage cubemap; // cube-compatible, mipmapped
	AllocatedImage irradiancemap;
	AllocatedImage prefilteredmap;
	AllocatedImage brdfLUT;
	VkSampler linearClampSampler{};

	// Graphics pipeline
	VkDescriptorSetLayout iblSetLayout{};
	VkDescriptorSet iblSet{}; // grow this later (irradiance, prefiltered, brdfLUT)

	// Equirect -> Cubemap compute pipeline
	VkDescriptorSetLayout convertSetLayout{};
	VkPipelineLayout convertPipelineLayout{};
	VkPipeline convertPipeline{};
	VkDescriptorSet convertSet{};

	// Radiance cubemap -> Irradiance cubemap compute pipeline
	VkDescriptorSetLayout irradianceSetLayout{};
	VkPipelineLayout irradiancePipelineLayout{};
	VkPipeline irradiancePipeline{};
	VkDescriptorSet irradianceSet{};

	// Roughness prefiltering compute pipeline
	VkDescriptorSetLayout prefilterSetLayout{};
	VkPipelineLayout prefilterPipelineLayout{};
	VkPipeline prefilterPipeline{};
	VkDescriptorSet prefilterSet{};

	// Roughness prefiltering compute pipeline
	VkDescriptorSetLayout brdfSetLayout{};
	VkPipelineLayout brdfPipelineLayout{};
	VkPipeline brdfPipeline{};
	VkDescriptorSet brdfSet{};
};

struct GPUShadowMapData {
	glm::mat4 lightSpaceMatrix;
};

struct ShadowMappingResources {
	DescriptorAllocatorGrowable descriptorAllocator;

	AllocatedImage shadowMap;
	VkSampler shadowSampler{};
	std::vector<VkImageView> layerViews;

	// Graphics pipeline
	VkDescriptorSetLayout shadowSetLayout{};
	VkDescriptorSet shadowSet{}; 

	VkDescriptorSetLayout shadowUboSetLayout{};
	VkDescriptorSet shadowUboSet{};

	// Shadow mapping prepass
	VkDescriptorSetLayout prepassSetLayout{};
	VkPipelineLayout prepassPipelineLayout{};
	VkPipeline prepassPipeline{};
	VkDescriptorSet prepassSet{};

	std::vector<GPUShadowMapData> dataPerCascade;

	uint32_t numCascades;
	std::vector<float> cascadePlanes;

	std::vector<glm::vec2> orthoDims;
};

struct PointLight {
	glm::vec4 pos_radius;
	glm::vec4 color_intensity;
};

struct LightParams {
	uint32_t lightCount;
	glm::uvec3 gridDim;
	uint32_t useClusters;
};

struct ClusterBuilderPushConstantsIn {
	float nearPlane;
	float farPlane;
};

struct ClusterBuilderIn {
	glm::mat4 invProj;
	glm::uvec4 tileSizes; // stores tile size in pixel in screenspace in .w component
	glm::uvec4 screenDimensions; // .yw for alignment padding
};

struct ClusterBuilderOut {
	glm::vec4 minPoint;
	glm::vec4 maxPoint;
};

struct ClusteredLightResources {
	DescriptorAllocatorGrowable descriptorAllocator;
	VkDescriptorSetLayout setLayout{};
	VkDescriptorSet set{};

	std::vector<PointLight> lights;
	std::vector<uint32_t> offsets;
	std::vector<uint32_t> indices;
	LightParams lightParams{};

	// --- Cluster builder ---
	// CPU Side data
	ClusterBuilderPushConstantsIn builderPC;
	ClusterBuilderIn builderIn;
	ClusterBuilderOut builderOut;
	uint32_t numClusters;
	glm::uvec3 tiles;
	// Pipeline + Descriptor
	VkDescriptorSetLayout builderSetLayout{};
	VkPipelineLayout builderPipelineLayout{};
	VkPipeline builderPipeline{};
	VkDescriptorSet builderSet{};
	
};

class VulkanEngine {
public:
	Camera mainCamera;
	double deltaTime;
	EngineStats stats;
	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
	float nearPlane;
	float farPlane;

	float mAzimuth = glm::radians(180.0f);
	float mZenith = glm::radians(20.0f);
	float lightDist = 90.0f;
	float lightIntensity = 4.0f;
	glm::vec3 lightColor = glm::vec3(1.0f, 0.8f, 0.8f);

	ClusteredLightResources _lightRes;

	ShadowMappingResources _shadowRes;

	IBLResources _ibl;
	VkPipelineLayout _skyboxPipelineLayout;
	VkPipeline _skyboxPipeline;

	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };
	struct SDL_Window* _window{ nullptr }; // forward declaration

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkSemaphore> _renderFinishedSemaphores;
	std::vector<VkImage> _swapchainImages; // Multiple frames in flight
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }; // Return ref. to cmdpool/buffer for current frame
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	VkExtent2D _drawExtent;
	float renderScale = 1.f;

	DescriptorAllocatorGrowable globalDescriptorAllocator;
	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;
	
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	std::vector<std::shared_ptr<MeshAsset>> testMeshes;

	bool resizeRequested;

	GPUSceneData sceneData;
	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	// textures initialized in init_default_data
	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;
	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;

	// combined texture + sampler
	VkDescriptorSetLayout _singleImageDescriptorLayout;

	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial;

	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes; // meshes
	
	// Functions

	static VulkanEngine& Get();
	void init();
	void cleanup();
	
	void draw();
	void run();

	AllocatedBuffer build_cluster_grid();

	void update_cluster_size(uint32_t tileSizePx, uint32_t numZSlices);

	void init_cluster_building_compute_pipeline();

	void init_point_light_descriptor_set();

	void init_shadow_mapping_pipeline();

	void init_shadow_mapping_descriptor_set();

	LightInfo compute_light_space_matrix(float, float, const glm::mat4& view, const glm::vec4& lightDir4);

	std::vector<LightInfo> getLightSpaceMatrices(uint32_t num_cascades, const std::vector<float>& cascade_planes, float near_plane, float far_plane, const glm::mat4& view, const glm::vec4& lightDir);

	void init_brdf_integration_pipeline();

	AllocatedImage generate_brdf_lut(uint32_t size, bool mipmapped);

	void init_prefiltered_cubemap_pipeline();

	AllocatedImage generate_prefiltered_map_from_cubemap(uint32_t cubeSize);

	void init_irradiance_cubemap_pipeline();

	AllocatedImage generate_irradiance_map_from_cubemap(uint32_t cubeSize, bool mipmapped);

	AllocatedImage create_cubemap_image(uint32_t size, VkFormat format, bool mipmapped);

	void init_equirect_to_cubemap_pipeline();

	AllocatedImage generate_cubemap_from_hdr(const char* hdrPath, uint32_t cubeSize, bool mipmapped);

	void init_ibl_descriptor_set();

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	void draw_geometry(VkCommandBuffer cmd);

	void resize_swapchain();

	void init_default_data();

	void init_mesh_pipeline();

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void addRandomPointLightsInRect(int N, glm::vec2 corner1, glm::vec2 corner2);
	VkImageView create_view(VkImage image, VkFormat format, VkImageAspectFlags aspect, VkImageViewType type, uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer, uint32_t layerCount);
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped, uint32_t arrayLayers, VkImageViewType viewType, VkImageCreateFlags flags = 0, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
	void destroy_image(const AllocatedImage& img);

	void update_scene();

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);
private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	void draw_background(VkCommandBuffer cmd);

	void init_descriptors();

	void init_pipelines();
	void init_background_pipelines();

	void init_skybox_pipeline();

	void init_imgui();
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
