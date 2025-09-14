#pragma once

#include <vk_descriptors.h>
#include <vk_types.h>

#include <unordered_map>
#include <filesystem>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

enum ImageUse : uint32_t { 
    None = 0, 
    BaseColor = 1 << 0, 
    Emissive = 1 << 1, 
    Normal = 1 << 2, 
    MetalRough = 1 << 3, 
    AO = 1 << 4 
};

struct GLTFMaterial {
	MaterialInstance data;
};

struct Bounds { // oriented bounding box for frustum culling
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

struct GeoSurface {
	uint32_t startIndex; // indexed draw offset
	uint32_t count;
	std::shared_ptr<GLTFMaterial> material;
    Bounds bounds;
};

struct MeshAsset {
    std::string name;
    std::vector<GeoSurface> surfaces; // array of sub-meshes
    GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);

struct LoadedGLTF : public IRenderable {
    // storage for all the data on a given glTF file
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes; // created from transform tree
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // nodes that dont have a parent, for iterating through the file in tree order
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    DescriptorAllocatorGrowable descriptorPool; // for materials of this gltf

    AllocatedBuffer materialDataBuffer; // single buffer containing all of the material data for the file

    VulkanEngine* creator;

    ~LoadedGLTF() { clearAll(); };

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);
private:
    void clearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath);
VkFilter extract_filter(fastgltf::Filter filter);
VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter);