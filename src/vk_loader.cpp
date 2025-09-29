#include <vk_loader.h>

#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

void LoadedGLTF::Draw(const glm::mat4& topMatrix, DrawContext& ctx) {
    // create renderables from the scenenodes
    for (auto& n : topNodes) {
        n->Draw(topMatrix, ctx);
    }
}

// If you want to destroy a LoadedGLTF at runtime, either do a VkQueueWait like we have in the cleanup function, 
// or add it into the per-frame deletion queue and defer it.
void LoadedGLTF::clearAll() {
    VkDevice dv = creator->_device;
    
    for (auto& [k, v] : meshes) {
        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto& [k, v] : images) {
        if (v.image == creator->_errorCheckerboardImage.image) {
            //dont destroy the default images
            continue;
        }
        creator->destroy_image(v);
    }

    for (auto& sampler : samplers) {
        vkDestroySampler(dv, sampler, nullptr);
    }

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);
}

inline bool is_srgb(VkFormat f) {
    return f == VK_FORMAT_R8G8B8A8_SRGB;
}

inline bool should_premultiply(ImageUse use) {
    return (uint32_t(use) & ImageUse::BaseColor) != 0;
}

// sRGB <-> linear helpers
struct SRGBLUT {
    float  s2l[256];
    uint8_t l2s[4097]; // 0..1 mapped to 0..4096 for table index
    SRGBLUT() {
        for (int i = 0;i < 256;++i) {
            float c = i / 255.f;
            s2l[i] = (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
        for (int i = 0;i <= 4096;++i) {
            float c = i / 4096.f;
            float s = (c <= 0.0031308f) ? (12.92f * c) : (1.055f * std::pow(c, 1.f / 2.4f) - 0.055f);
            l2s[i] = (uint8_t)std::round(std::clamp(s, 0.f, 1.f) * 255.f);
        }
    }
};
inline const SRGBLUT& srgb_lut() { static SRGBLUT L; return L; }

inline uint8_t lin_to_srgb_8(float lin) {
    const auto& L = srgb_lut();
    int idx = (int)std::round(std::clamp(lin, 0.f, 1.f) * 4096.f);
    return L.l2s[idx];
}

inline float srgb_to_lin_8(uint8_t s) {
    return srgb_lut().s2l[s];
}

void premultiply_rgba_inplace(uint8_t* data, int w, int h, bool storageIsSRGB) {
    const size_t n = size_t(w) * size_t(h);
    if (storageIsSRGB) { // Convert RGB sRGB->linear, multiply by alpha, convert back to sRGB
        for (size_t i = 0;i < n;++i) {
            uint8_t* px = data + i * 4;
            float a = px[3] / 255.f;
            float r = srgb_to_lin_8(px[0]) * a;
            float g = srgb_to_lin_8(px[1]) * a;
            float b = srgb_to_lin_8(px[2]) * a;
            px[0] = lin_to_srgb_8(r);
            px[1] = lin_to_srgb_8(g);
            px[2] = lin_to_srgb_8(b);
            px[3] = px[3]; // alpha stays linear as 8-bit UNORM:
        }
    }
    else { // Linear UNORM storage; just scale RGB by alpha
        for (size_t i = 0;i < n;++i) {
            uint8_t* px = data + i * 4;
            float a = px[3] / 255.f;
            px[0] = (uint8_t)std::round(px[0] * a);
            px[1] = (uint8_t)std::round(px[1] * a);
            px[2] = (uint8_t)std::round(px[2] * a);
            // alpha unchanged
        }
    }
}

std::optional<AllocatedImage> load_image(
    VulkanEngine* engine,
    VkFormat format,
    ImageUse use,
    fastgltf::Asset& asset,
    fastgltf::Image& image,
    const std::filesystem::path& base_dir)
{
    AllocatedImage newImage{};
    int width = 0, height = 0, nrChannels = 0;

    auto upload = [&](unsigned char* data) {
        if (!data) { fmt::println("stbi_load failed"); return; }

        if (should_premultiply(use)) {
            premultiply_rgba_inplace(data, width, height, is_srgb(format));
        }

        VkExtent3D imagesize{ (uint32_t)width, (uint32_t)height, 1 };
        newImage = engine->create_image(data, imagesize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
        stbi_image_free(data);
    };

    std::visit(
        fastgltf::visitor{
            [](auto&) {},
            [&](fastgltf::sources::URI& filePath) {
                assert(filePath.fileByteOffset == 0);
                assert(filePath.uri.isLocalPath());
                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                upload(data);
            },
            [&](fastgltf::sources::Vector& vector) {
                unsigned char* data = stbi_load_from_memory(vector.bytes.data(),
                                                            (int)vector.bytes.size(),
                                                            &width,&height,&nrChannels,4);
                upload(data);
            },
            [&](fastgltf::sources::BufferView& view) {
                auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                auto& buffer = asset.buffers[bufferView.bufferIndex];
                std::visit(fastgltf::visitor{
                    [](auto&) {},
                    [&](fastgltf::sources::Vector& vector) {
                        unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset,
                                                                    (int)bufferView.byteLength,
                                                                    &width,&height,&nrChannels,4);
                        upload(data);
                    }
                }, buffer.data);
            }
        }, image.data);

    if (newImage.image == VK_NULL_HANDLE) return {};
    return newImage;
}

std::optional<AllocatedImage> load_image(VulkanEngine* engine, VkFormat format, fastgltf::Asset& asset, fastgltf::Image& image, const std::filesystem::path& base_dir) {
    AllocatedImage newImage{};

    int width, height, nrChannels;

    std::visit(
        fastgltf::visitor{
            [](auto& arg) {},
            [&](fastgltf::sources::URI& filePath) {
                assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.
                fmt::println("fastgltf::sources::URI& filePath");
                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                fmt::println("Loading from {}", path);
                unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if (data) {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;
                    newImage = engine->create_image(data, imagesize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
                    stbi_image_free(data);
                }
                else {
                    fmt::println("stbi_load failed");
                }
            },

            [&](fastgltf::sources::Vector& vector) {
                fmt::println("fastgltf::sources::Vector& vector");
                unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                if (data) {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;
                    newImage = engine->create_image(data, imagesize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
                    stbi_image_free(data);
                }
                else {
                    fmt::println("stbi_load failed");
                }
            },

            [&](fastgltf::sources::BufferView& view) {
                fmt::println("fastgltf::sources::BufferView& view");
                auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                auto& buffer = asset.buffers[bufferView.bufferIndex];
                std::visit(fastgltf::visitor { 
                    // We only care about VectorWithMime here, because we
                    // specify LoadExternalBuffers, meaning all buffers
                    // are already loaded into a vector.
                    [](auto& arg) {},
                    [&](fastgltf::sources::Vector& vector) {
                        unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), 
                            &width, &height, &nrChannels, 4);
                        if (data) {
                            VkExtent3D imagesize;
                            imagesize.width = width;
                            imagesize.height = height;
                            imagesize.depth = 1;
                            newImage = engine->create_image(data, imagesize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
                            stbi_image_free(data);
                        }
                        else {
                            fmt::println("stbi_load failed");
                        }
                    } 
                }, buffer.data);
            },
        }, image.data);

    // if any of the attempts to load the data failed, we havent written the image
    // so handle is null
    if (newImage.image == VK_NULL_HANDLE) {
        return {};
    }
    else {
        return newImage;
    }
}

ImageUse operator|(ImageUse a, ImageUse b) {
    return ImageUse(uint32_t(a) | uint32_t(b));
}

ImageUse& operator|=(ImageUse& a, ImageUse b) {
    a = a | b; return a;
}

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath) {
    fmt::println("Loading GLTF: {}", filePath);

    const std::filesystem::path gltf_fs_path(filePath);
    const std::filesystem::path base_dir = gltf_fs_path.parent_path();

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF& file = *scene.get();

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember 
        | fastgltf::Options::AllowDouble 
        | fastgltf::Options::LoadGLBBuffers 
        | fastgltf::Options::LoadExternalBuffers
        | fastgltf::Options::LoadExternalImages;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    // Load file (glTF or GLB)
    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF) {
        auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        }
        else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB) {
        auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        }
        else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else {
        std::cerr << "Failed to determine glTF container" << std::endl;
        return {};
    }

    // Create descriptor pool mgr, use gltf to estimate number of initial sets
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = { 
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } 
    };
    fmt::println("gltf.materials.size() = {}", gltf.materials.size());
    file.descriptorPool.init(engine->_device, gltf.materials.size(), sizes);

    // load samplers
    // load samplers
    if (gltf.samplers.empty()) {
        fmt::println("No samplers in glTF; creating default sampler");

        VkSamplerCreateInfo sampl = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr
        };
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;
        sampl.magFilter = VK_FILTER_LINEAR;              // default mag filter
        sampl.minFilter = VK_FILTER_LINEAR;              // default min filter
        sampl.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;// good default
        sampl.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampl.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampl.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &sampl, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }
    else {
        int sampler_idx = 0;
        for (fastgltf::Sampler& sampler : gltf.samplers) {
            fmt::println("Loading sampler {}", sampler_idx);

            VkSamplerCreateInfo sampl = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr
            };
            sampl.maxLod = VK_LOD_CLAMP_NONE;
            sampl.minLod = 0;
            sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
            sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
            sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

            VkSampler newSampler;
            vkCreateSampler(engine->_device, &sampl, nullptr, &newSampler);

            file.samplers.push_back(newSampler);
            sampler_idx++;
        }
    }

    // temporal arrays for all the objects to use while creating the GLTF data
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    // Textures > Materials > Meshes > MeshNodes

    std::vector<ImageUse> usage(gltf.images.size(), ImageUse::None);

    // helper: tag by texture index -> image index
    auto tag_by_tex_index = [&](size_t texIndex, ImageUse u) {
        if (texIndex >= gltf.textures.size()) return;
        const auto& tex = gltf.textures[texIndex];
        if (!tex.imageIndex) return; // skip if no image (e.g., invalid)
        size_t img = tex.imageIndex.value();
        if (img >= usage.size()) return;
        usage[img] |= u;
    };

    // generic helper: works for std::optional<fastgltf::TextureInfo>,
    // std::optional<fastgltf::NormalTextureInfo>, std::optional<fastgltf::OcclusionTextureInfo>, etc.
    auto tag_tex = [&](const auto& optTexInfo, ImageUse u) {
        if (!optTexInfo.has_value()) return;
        tag_by_tex_index(optTexInfo->textureIndex, u);
    };

    // pass 1: tag usage
    for (auto& mat : gltf.materials) {
        tag_tex(mat.pbrData.baseColorTexture, ImageUse::BaseColor);
        tag_tex(mat.pbrData.metallicRoughnessTexture, ImageUse::MetalRough);
        tag_tex(mat.normalTexture, ImageUse::Normal); // NormalTextureInfo
        tag_tex(mat.occlusionTexture, ImageUse::AO); // OcclusionTextureInfo
        tag_tex(mat.emissiveTexture, ImageUse::Emissive);
    }

    // choose format
    auto chooseFormat = [&](ImageUse u) -> VkFormat {
        bool wantsSRGB = (uint32_t(u) & (ImageUse::BaseColor | ImageUse::Emissive)) != 0;
        return wantsSRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    };

    // load all textures
    for (size_t img_idx = 0; img_idx < gltf.images.size(); ++img_idx) {
        fastgltf::Image& image = gltf.images[img_idx];
        VkFormat fmt = chooseFormat(usage[img_idx]);
        std::optional<AllocatedImage> img = load_image(engine, fmt, usage[img_idx], gltf, image, base_dir); // add param

        if (img.has_value()) {
            images.push_back(*img);
            fmt::println("Inserting into file.images[{}]", std::to_string(img_idx));
            file.images[std::to_string(img_idx)] = *img;
        }
        else {
            // we failed to load, so lets give the slot a default white texture to not
            // completely break loading
            images.push_back(engine->_errorCheckerboardImage);
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
    }

    // allocate buffer for material data with VMA
    file.materialDataBuffer = engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // pointer to buffer we just allocated. Used for writes below
    GLTFMetallic_Roughness::MaterialConstants* sceneMaterialConstants = (GLTFMetallic_Roughness::MaterialConstants*)file.materialDataBuffer.info.pMappedData;
    int mat_idx = 0;
    for (fastgltf::Material& mat : gltf.materials) {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        fmt::println("Inserting into file.materials[{}]", std::to_string(mat_idx));
        file.materials[std::to_string(mat_idx)] = newMat;

        GLTFMetallic_Roughness::MaterialConstants constants;
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
        constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;

        constants.emissiveFactors.x = mat.emissiveFactor[0];
        constants.emissiveFactors.y = mat.emissiveFactor[1];
        constants.emissiveFactors.z = mat.emissiveFactor[2];

        // write material parameters to buffer
        sceneMaterialConstants[mat_idx] = constants;

        // determine if it's opaque or transparent based on material blending type
        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend) {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_Roughness::MaterialResources materialResources;

        // fallbacks
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;
        materialResources.normalImage = engine->_whiteImage;
        materialResources.normalSampler = engine->_defaultSamplerLinear;
        materialResources.occlusionImage = engine->_whiteImage;
        materialResources.occlusionSampler = engine->_defaultSamplerLinear;
        materialResources.emissiveImage = engine->_whiteImage;
        materialResources.emissiveSampler = engine->_defaultSamplerLinear;

        // set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = mat_idx * sizeof(GLTFMetallic_Roughness::MaterialConstants);

        // base color
        if (mat.pbrData.baseColorTexture.has_value()) {
            fmt::println("Loading base color texture");
            size_t texIndex = mat.pbrData.baseColorTexture.value().textureIndex;
            size_t img = gltf.textures[texIndex].imageIndex.value();
            size_t sampler = gltf.textures[texIndex].samplerIndex.value_or(0); 
            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }

        // metalness + roughness
        if (mat.pbrData.metallicRoughnessTexture.has_value()) {
            fmt::println("Loading metal+roughness texture");
            size_t texIndex = mat.pbrData.metallicRoughnessTexture.value().textureIndex;
            size_t img = gltf.textures[texIndex].imageIndex.value();
            size_t sampler = gltf.textures[texIndex].samplerIndex.value_or(0);
            materialResources.metalRoughImage = images[img];
            materialResources.metalRoughSampler = file.samplers[sampler];
        }

        // normal
        if (mat.normalTexture.has_value()) {
            fmt::println("Loading normal texture");
            size_t texIndex = mat.normalTexture.value().textureIndex;
            size_t img = gltf.textures[texIndex].imageIndex.value();
            size_t sampler = gltf.textures[texIndex].samplerIndex.value_or(0);
            materialResources.normalImage = images[img];
            materialResources.normalSampler = file.samplers[sampler];
        }

        // occlusion
        if (mat.occlusionTexture.has_value()) {
            fmt::println("Loading AO texture");
            size_t texIndex = mat.occlusionTexture.value().textureIndex;
            size_t img = gltf.textures[texIndex].imageIndex.value();
            size_t sampler = gltf.textures[texIndex].samplerIndex.value_or(0);
            materialResources.occlusionImage = images[img];
            materialResources.occlusionSampler = file.samplers[sampler];
        }

        // emissive
        if (mat.emissiveTexture.has_value()) {
            fmt::println("Loading emissive texture");
            size_t texIndex = mat.emissiveTexture.value().textureIndex;
            size_t img = gltf.textures[texIndex].imageIndex.value();
            size_t sampler = gltf.textures[texIndex].samplerIndex.value_or(0);
            materialResources.emissiveImage = images[img];
            materialResources.emissiveSampler = file.samplers[sampler];
        }

        // write descriptors
        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources, file.descriptorPool);
        newMat->zPrepassData = engine->metalRoughMaterial.write_z_prepass_material(engine->_device, passType, materialResources, file.descriptorPool);
        newMat->geometryPrepassData = engine->metalRoughMaterial.write_geometry_prepass_material(engine->_device, passType, materialResources, file.descriptorPool, newMat->data.materialSet);

        mat_idx++;
    }

    // use the same vectors for all meshes so that the memory doesnt reallocate as often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    int mesh_idx = 0;
    for (fastgltf::Mesh& mesh : gltf.meshes) {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        fmt::println("Inserting into file.meshes[{}]", std::to_string(mesh_idx));
        file.meshes[std::to_string(mesh_idx)] = newmesh;
        newmesh->name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initial_vtx);
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 }; // defaults? replaced below
                        newvtx.color = glm::vec4{ 1.f };
                        newvtx.uv_x = 0;
                        newvtx.uv_y = 0;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v; // here
                    });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                    [&](glm::vec2 v, size_t index) {
                        vertices[initial_vtx + index].uv_x = v.x; // same
                        vertices[initial_vtx + index].uv_y = v.y;
                    });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) {
                        vertices[initial_vtx + index].color = v;
                    });
            }

            if (p.materialIndex.has_value()) {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else {
                newSurface.material = materials[0];
            }

            // loop the vertices of this surface, find min/max bounds
            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (int i = initial_vtx; i < vertices.size(); i++) {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }
            newSurface.bounds.origin = (maxpos + minpos) / 2.f; // really just the midpoint of min/max
            newSurface.bounds.extents = (maxpos - minpos) / 2.f; // from the midpoint to min/max (it is symmetric)
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->uploadMesh(indices, vertices);

        mesh_idx++;
    }

    // load all nodes and their meshes
    int node_idx = 0;
    for (fastgltf::Node& node : gltf.nodes) {
        std::shared_ptr<Node> newNode;

        // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode class
        if (node.meshIndex.has_value()) {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode*>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        fmt::println("Inserting into file.nodes[{}]", std::to_string(node_idx));
        file.nodes[std::to_string(node_idx)];

        // If node.transform holds a TransformMatrix, the first lambda runs
        // If it holds a TRS, the second lambda runs
        std::visit(fastgltf::visitor{ // fastgltf::visitor internally creates a callable object with an operator() overload for each lambda
            [&](fastgltf::Node::TransformMatrix matrix) {
                memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix)); // just copy the matrix if that's what the node uses
            }, 
            [&](fastgltf::Node::TRS transform) { 
                glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);
                glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                glm::mat4 rm = glm::toMat4(rot);
                glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);
                newNode->localTransform = tm * rm * sm; // otherwise construct it from the TRS
            } 
        }, node.transform); // .transform is a variant, between either a matrix or a TRS

        node_idx++;
    }

    // Nodes loaded, now construct hierarchy to create scene graph
    // run loop again to setup transform hierarchy
    for (int i = 0; i < gltf.nodes.size(); i++) {
        fastgltf::Node& node = gltf.nodes[i];
        std::shared_ptr<Node>& sceneNode = nodes[i];

        for (auto& c : node.children) {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes, with no parents
    for (auto& node : nodes) {
        if (node->parent.lock() == nullptr) {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{ 1.f });
        }
    }

    return scene;
}

// GL to VK conversion
VkFilter extract_filter(fastgltf::Filter filter) {
    switch (filter) {
        // nearest samplers
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

        // linear samplers
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_FILTER_LINEAR;
    }
}

// GL to VK conversion
VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter) {
    switch (filter) {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
{
    std::cout << "Loading GLTF: " << filePath << std::endl;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers
        | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
    if (load) {
        gltf = std::move(load.get());
    }
    else {
        fmt::print("Failed to load glTF: {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }

    // loop each mesh
    // copy vertices/indices into temporary vector
    // upload them as mesh to engine

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    // use the same vectors for all meshes so that the memory doesnt reallocate as
    // often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : gltf.meshes) {
        MeshAsset newmesh;

        newmesh.name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initial_vtx);
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 };
                        newvtx.color = glm::vec4{ 1.f };
                        newvtx.uv_x = 0;
                        newvtx.uv_y = 0;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                    [&](glm::vec2 v, size_t index) {
                        vertices[initial_vtx + index].uv_x = v.x;
                        vertices[initial_vtx + index].uv_y = v.y;
                    });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) {
                        vertices[initial_vtx + index].color = v;
                    });
            }
            newmesh.surfaces.push_back(newSurface);
        }

        // display the vertex normals
        constexpr bool OverrideColors = false;
        if (OverrideColors) {
            for (Vertex& vtx : vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }
        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }

    return meshes;
}

// IBL

AllocatedImage create_equirect_image_from_hdr(VulkanEngine* engine, const char* path) {
    int w, h, comp;
    float* data = stbi_loadf(path, &w, &h, &comp, 4); // RGB32F
    if (!data) {
        fmt::println("Failed to load equirectangular image at {}", path);
    }

    VkExtent3D ext{ 
        (uint32_t)w, 
        (uint32_t)h, 
        1 
    };

    // sampled by compute shader
    return engine->create_image(data, ext, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
}

