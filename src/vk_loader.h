﻿#pragma once

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>

// sub-mesh
// we'll use start index as offsets for subsequent draw calls into the same buffer
struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
};

struct MeshAsset {
    std::string name;
    std::vector<GeoSurface> surfaces; // array of sub-meshes
    GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);