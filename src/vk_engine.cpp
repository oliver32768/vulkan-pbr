#include "vk_engine.h"

#include "VkBootstrap.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtc/epsilon.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include <vk_pipelines.h>

#include <chrono>
#include <thread>
#include <random>

#ifdef _DEBUG
constexpr bool bUseValidationLayers = true;
#else
constexpr bool bUseValidationLayers = false;
#endif

uint32_t MAX_MIPS = 15;

VulkanEngine* loadedEngine = nullptr;

void VulkanEngine::addRandomPointLightsInRect(int N, glm::vec2 corner1, glm::vec2 corner2) {
    // corner components are (x,z)
    const float minX = std::min(corner1.x, corner2.x);
    const float maxX = std::max(corner1.x, corner2.x);
    const float minZ = std::min(corner1.y, corner2.y);
    const float maxZ = std::max(corner1.y, corner2.y);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distX(minX, maxX);
    std::uniform_real_distribution<float> distZ(minZ, maxZ);
    std::uniform_real_distribution<float> distY(0.2f, 0.8f);
    std::uniform_real_distribution<float> distRadius(0.5f, 10.0f);
    std::uniform_real_distribution<float> distIntensity(5.0f, 10.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    _lightRes.lights.reserve(_lightRes.lights.size() + static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) {
        // Random unit RGB (normalize random vector from [0,1]^3)
        glm::vec3 rgb(dist01(gen), dist01(gen), dist01(gen));
        float m = glm::length(rgb);
        if ((m * m) < 1e-10f) rgb = glm::vec3(1, 0, 0);
        rgb = glm::normalize(rgb);

        PointLight l{};
        l.pos_radius = glm::vec4(
            distX(gen), // x (independent)
            distY(gen), // y in [0.2, 0.8]
            distZ(gen), // z (independent)
            distRadius(gen) // radius in [0.5, 10]
        );

        l.color_intensity = glm::vec4(
            rgb, // random unit color
            distIntensity(gen) // intensity in [5, 10]
        );

        _lightRes.lights.push_back(l);
    }
}

// lambda = 0 => uniform, lambda = 1 => logarithmic. Typically 0.6-0.9
static std::vector<float> buildCascadePlanes(int numCascades, float nearPlane, float farPlane, float lambda) {
    std::vector<float> planes(numCascades + 1);
    planes[0] = nearPlane;

    for (int i = 1; i < numCascades; ++i) {
        float si = float(i) / float(numCascades);
        float uni = nearPlane + (farPlane - nearPlane) * si; // uniform
        float log = nearPlane * std::pow(farPlane / nearPlane, si); // logarithmic
        planes[i] = glm::mix(uni, log, lambda); // blend
    }

    planes[numCascades] = farPlane;
    return planes;
}

static std::array<glm::vec3, 8> frustumCornersWS_fromView(const glm::mat4& view, float fovY_rad, float aspect, float n, float f)
{
    // view -> world
    glm::mat4 invView = glm::inverse(view);

    float tanHalf = std::tan(fovY_rad * 0.5f);
    float nh = n * tanHalf;
    float nw = nh * aspect;
    float fh = f * tanHalf;
    float fw = fh * aspect;

    // View-space corners (Right-Handed, camera looks -Z)
    std::array<glm::vec3, 8> vs = {
        glm::vec3(-nw,  nh, -n), glm::vec3(nw,  nh, -n),
        glm::vec3(nw, -nh, -n), glm::vec3(-nw, -nh, -n),
        glm::vec3(-fw,  fh, -f), glm::vec3(fw,  fh, -f),
        glm::vec3(fw, -fh, -f), glm::vec3(-fw, -fh, -f),
    };

    std::array<glm::vec3, 8> ws;
    for (int i = 0; i < 8; ++i) {
        glm::vec4 w = invView * glm::vec4(vs[i], 1.0f);
        ws[i] = glm::vec3(w);
    }
    return ws;
}

static inline bool is_depth_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default: return false;
    }
}

static inline bool has_stencil(VkFormat f) {
    switch (f) {
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default: return false;
    }
}

static VkImageAspectFlags aspect_for_format(VkFormat format) {
    if (!is_depth_format(format)) return VK_IMAGE_ASPECT_COLOR_BIT;
    return has_stencil(format) ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) 
        : VK_IMAGE_ASPECT_DEPTH_BIT;
}

// Create an image-view with full control over type/layers/mips
VkImageView VulkanEngine::create_view(VkImage image, VkFormat format, VkImageAspectFlags aspect,
    VkImageViewType type,
    uint32_t baseMip, uint32_t mipCount,
    uint32_t baseLayer, uint32_t layerCount)
{
    VkImageViewCreateInfo vi = vkinit::imageview_create_info(format, image, aspect);
    vi.viewType = type;
    vi.subresourceRange.baseMipLevel = baseMip;
    vi.subresourceRange.levelCount = mipCount;
    vi.subresourceRange.baseArrayLayer = baseLayer;
    vi.subresourceRange.layerCount = layerCount;

    VkImageView view;
    VK_CHECK(vkCreateImageView(_device, &vi, nullptr, &view));
    return view;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size,
    VkFormat format,
    VkImageUsageFlags usage,
    bool mipmapped,
    uint32_t arrayLayers,
    VkImageViewType viewType,
    VkImageCreateFlags flags /*=0*/,
    VkSampleCountFlagBits samples /*=VK_SAMPLE_COUNT_1_BIT*/)
{
    AllocatedImage img{};
    img.imageFormat = format;
    img.imageExtent = size;

    // Base image info
    VkImageCreateInfo ii = vkinit::image_create_info(format, usage, size);
    ii.flags = flags;
    ii.samples = samples;
    ii.arrayLayers = std::max(1u, arrayLayers);
    ii.imageType = (size.depth > 1 && viewType == VK_IMAGE_VIEW_TYPE_3D) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; // as before

    if (mipmapped) {
        const uint32_t maxDim = std::max(size.width, size.height);
        ii.mipLevels = std::max(1u, static_cast<uint32_t>(std::floor(std::log2(maxDim))) + 1);
    }

    // Allocate with VMA
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VK_CHECK(vmaCreateImage(_allocator, &ii, &aci, &img.image, &img.allocation, nullptr));

    // Aspect + default view (spans all mips and all array layers)
    const VkImageAspectFlags aspect = aspect_for_format(format);
    img.imageView = create_view(img.image, format, aspect, viewType, 0, ii.mipLevels, 0, ii.arrayLayers);

    return img;
}

// Frustum culling
bool is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    // Local AABB corners
    const glm::vec3 c[8] = {
        { 1,  1,  1}, { 1,  1, -1}, { 1, -1,  1}, { 1, -1, -1},
        {-1,  1,  1}, {-1,  1, -1}, {-1, -1,  1}, {-1, -1, -1},
    };

    glm::mat4 M = obj.transform;
    glm::mat4 VP = viewproj;

    glm::vec4 v[8];
    for (int i = 0; i < 8; ++i) {
        glm::vec3 p = obj.bounds.origin + (c[i] * obj.bounds.extents); // local AABB -> local point
        v[i] = VP * (M * glm::vec4(p, 1.0f)); // world -> clip
    }

    auto outside_all = [&](auto pred) {
        for (int i = 0; i < 8; ++i) if (!pred(v[i])) return false;
        return true;
    };

    // Left
    if (outside_all([](const glm::vec4& q) { return q.x < -q.w; })) return false;
    if (outside_all([](const glm::vec4& q) { return q.x > q.w; })) return false;

    // Bottom
    if (outside_all([](const glm::vec4& q) { return q.y < -q.w; })) return false;
    if (outside_all([](const glm::vec4& q) { return q.y > q.w; })) return false;

    // Near/Far depend on clip-space convention:
    if (outside_all([](const glm::vec4& q) { return q.z < 0.0f; })) return false; // near
    if (outside_all([](const glm::vec4& q) { return q.z > q.w;   })) return false; // far

    // Not outside any plane
    return true;
}

void VulkanEngine::destroy_image(const AllocatedImage& img) {
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

// takes ptr to img data and actually copies to via temporal staging buffer / immediate submit
AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped) {
    size_t pixel_size = 0;
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
            pixel_size = 4;  // 1 byte × 4
            break;
        case VK_FORMAT_R8G8B8A8_SRGB:
            pixel_size = 4;
            break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            pixel_size = 8;
            break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            pixel_size = 16; // 4 bytes × 4
            break;
        default:
            throw std::runtime_error("Unsupported format in create_image");
    }

    size_t data_size = size.depth * size.width * size.height * pixel_size;

    AllocatedBuffer uploadbuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        if (mipmapped) {
            vkutil::generate_mipmaps(cmd, new_image.image, VkExtent2D{ new_image.imageExtent.width,new_image.imageExtent.height });
        }
        else {
            vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
    destroy_buffer(uploadbuffer);
    return new_image;
}

// no actual data, just allocates
AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped) {
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // Use VMA to alloc img on GPU
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

void VulkanEngine::resize_swapchain() {
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resizeRequested = false;
}

void VulkanEngine::init_default_data() {
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void*)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = create_image((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = create_image((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16 > pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    // Samplers
    VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);
    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    // Default materials
    GLTFMetallic_Roughness::MaterialResources materialResources;
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;
    materialResources.normalImage = _whiteImage;
    materialResources.normalSampler = _defaultSamplerLinear;
    materialResources.occlusionImage = _whiteImage;
    materialResources.occlusionSampler = _defaultSamplerLinear;
    materialResources.emissiveImage = _blackImage;
    materialResources.emissiveSampler = _defaultSamplerLinear;

    // ubo
    AllocatedBuffer materialConstants = create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // write
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants*)materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{ 1,1,1,1 };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1, 0.5, 0, 0 };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
    });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
    });
}

void VulkanEngine::init_mesh_pipeline() {
    VkShaderModule triangleFragShader;
	if (!vkutil::load_shader_module("../../shaders/tex_image.frag.spv", _device, &triangleFragShader)) {
		fmt::print("Error when building the fragment shader \n");
	}
	else {
		fmt::print("Triangle fragment shader succesfully loaded \n");
	}

	VkShaderModule triangleVertexShader;
	if (!vkutil::load_shader_module("../../shaders/colored_triangle_mesh.vert.spv", _device, &triangleVertexShader)) {
		fmt::print("Error when building the vertex shader \n");
	}
	else {
		fmt::print("Triangle vertex shader succesfully loaded \n");
	}

	VkPushConstantRange bufferRange{};
	bufferRange.offset = 0;
	bufferRange.size = sizeof(GPUDrawPushConstants);
	bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
	pipeline_layout_info.pPushConstantRanges = &bufferRange;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
	pipeline_layout_info.setLayoutCount = 1;
	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;
    pipelineBuilder._pipelineLayout = _meshPipelineLayout; // use the triangle layout we created
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader); // connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST); // it will draw triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL); // filled triangles
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE); // no backface culling
    pipelineBuilder.set_multisampling_none(); // no multisampling
    pipelineBuilder.disable_blending(); // no blending
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL); // yes depth testing
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat); // connect the image format we will draw into, from draw image
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);
    _meshPipeline = pipelineBuilder.build_pipeline(_device); // finally build the pipeline

    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
    });
}

// uses cpu visible staging buffer and then copies that to gpu exclusive memory. returns struct containing vtx/idx ssbos and vtx ssbo address
// TODO: this pattern is not very efficient, as we are waiting for the GPU command to fully execute before continuing with our CPU side logic. 
// This is something people generally put on a background thread, whose sole job is to execute uploads like this one, and deleting/reusing the staging buffers.
GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer = create_buffer(
        vertexBufferSize, 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    // storage_buffer = ssbo
    // shader_device_address for pointer thingy
    // transfer_dst so we can copy to it

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo); // "ptr"

    //create index buffer
    newSurface.indexBuffer = create_buffer(
        indexBufferSize, 
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();
    memcpy(data, vertices.data(), vertexBufferSize); // copy vertex buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize); // copy index buffer

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

// 1. VMA_MEMORY_USAGE_GPU_ONLY: Fast but does not support R/W from host. Lives in GPU memory
// 2. VMA_MEMORY_USAGE_CPU_ONLY: Slow but supports R/W from CPU. Lives in host memory
// 3. VMA_MEMORY_USAGE_CPU_TO_GPU: Faster CPU writable memory. Lives in host visible region of GPU (typically small unless using ReBAR)
// 4. VMA_MEMORY_USAGE_GPU_TO_CPU: Used on memory that we want to be safely readable from CPU.
// Sidenote, writing from CPU will be done via mapped pointers (requires VMA_ALLOCATION_CREATE_MAPPED_BIT)
AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    // allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer) {
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) {
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo); // begin render pass - now we can execute draw cmds

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd); // imgui will records its commands into `cmd` (I guess this state comes from ImGui::ShowDemoWindow?)

    vkCmdEndRendering(cmd);
}

void VulkanEngine::init_imgui() {
    // sizes copied from imgui demo (stores 1000 copies of each type of descriptor)
    // TODO: apparently this is overkill
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool; // imgui wants its own descriptor pool
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    // dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat; // imgui draws directly into swapchain

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

// TODO: run this on a different queue than the graphics queue, that way we could overlap the execution from this with the main render loop
void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) {
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::init_background_pipelines() {
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // compute, vertex, frag, etc.

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = gradientShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    // default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    // change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    // default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    // add the 2 ComputeEffects into the array
    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    // destroy structures properly
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    });
}

void VulkanEngine::init_skybox_pipeline() {
    VkShaderModule skyboxFragShader;
    if (!vkutil::load_shader_module("../../shaders/skybox.frag.spv", _device, &skyboxFragShader)) {
        fmt::print("Error when building the fragment shader \n");
    }
    else {
        fmt::print("Skybox fragment shader succesfully loaded \n");
    }

    VkShaderModule skyboxVertexShader;
    if (!vkutil::load_shader_module("../../shaders/skybox.vert.spv", _device, &skyboxVertexShader)) {
        fmt::print("Error when building the vertex shader \n");
    }
    else {
        fmt::print("Skybox vertex shader succesfully loaded \n");
    }

    VkDescriptorSetLayout layouts[] = { _gpuSceneDataDescriptorLayout, _ibl.iblSetLayout };
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.setLayoutCount = 2;
    pipeline_layout_info.pSetLayouts = layouts;
    pipeline_layout_info.pPushConstantRanges = nullptr;
    pipeline_layout_info.pushConstantRangeCount = 0;
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_skyboxPipelineLayout));

    PipelineBuilder pipelineBuilder;
    pipelineBuilder._pipelineLayout = _skyboxPipelineLayout; // use the triangle layout we created
    pipelineBuilder.set_shaders(skyboxVertexShader, skyboxFragShader); // connecting the vertex and fragment shaders to the pipeline
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST); // it will draw triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL); // filled triangles
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE); // no backface culling
    pipelineBuilder.set_multisampling_none(); // no multisampling
    pipelineBuilder.disable_blending(); // no blending
    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL); // yes depth testing, but no depth writes
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat); // connect the image format we will draw into, from draw image
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);
    _skyboxPipeline = pipelineBuilder.build_pipeline(_device); // finally build the pipeline

    vkDestroyShaderModule(_device, skyboxFragShader, nullptr);
    vkDestroyShaderModule(_device, skyboxVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _skyboxPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _skyboxPipeline, nullptr);
     });
}

void VulkanEngine::init_pipelines() {
    init_background_pipelines();
    init_mesh_pipeline();
    metalRoughMaterial.build_pipelines(this);
    init_skybox_pipeline();
}

void VulkanEngine::init_descriptors() {
    // draw img descriptor for comp. shader
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
    };
    globalDescriptorAllocator.init(_device, 10, sizes);
    {
        DescriptorLayoutBuilder builder; // vk_descriptors.h
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);
    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, _drawImageDescriptors);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }

    // this ubo will be alloc'd every frame in `draw` but realistically this is only necessary for certain cases with dynamic drawing
    // it is usually better to just cache this in e.g. FrameData in our case and not realloc 
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // descriptor set with a single combined image sampler
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // make sure both the descriptor allocator and the new layout get cleaned up properly
    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::init_commands() {
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));
    _mainDeletionQueue.push_function([=]() { // TODO: Is capture by value really necessary here (this is not the only place this occurs)
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures() {
    // create syncronization structures
    // one fence to control when the gpu has finished rendering the frame,
    // and 2 semaphores to syncronize rendering with swapchain
    // we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));   
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
    }

    _renderFinishedSemaphores.resize(_swapchainImages.size());
    for (int i = 0; i < _swapchainImages.size(); i++) {
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderFinishedSemaphores[i]));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::destroy_swapchain() {
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    for (int i = 0; i < _swapchainImageViews.size(); i++) {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

// present mode (e.g. vsync) is defined at swapchain creation
void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device,_surface };

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // use vsync present mode
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::init_swapchain() {
    create_swapchain(_windowExtent.width, _windowExtent.height);

    int display = SDL_GetWindowDisplayIndex(_window);
    SDL_DisplayMode dm{};
    if (display >= 0 && SDL_GetDesktopDisplayMode(display, &dm) == 0) {
        int desktopW = dm.w;
        int desktopH = dm.h;
    }

    VkExtent3D drawImageExtent = {
        dm.w, // _windowExtent.width,
        dm.h, // _windowExtent.height,
        1
    };

    // hardcoding the draw format to 16 bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    //for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    //allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
    });
}

void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;

    //make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Example Vulkan Application")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    //grab the instance 
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = true; // dynamic rendering allows us to completely skip renderpasses/framebuffer
    features.synchronization2 = true;

    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true; // Buffer device adress will let us use GPU pointers without binding buffers
    features12.descriptorIndexing = true; // descriptorIndexing gives us bindless textures

    //use vkbootstrap to select a gpu. 
    //We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize VMA
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // allows for use of GPU pointers
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
    });
}

VulkanEngine& VulkanEngine::Get() { 
    return *loadedEngine; 
}

void VulkanEngine::init() {
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();

    init_equirect_to_cubemap_pipeline();
    // kloofendal_48d_partly_cloudy_puresky_4k
    // moon_lab_4k
    // empty_play_room_4k
    // brown_photostudio_02_4k
    // snow
    // metro
    // winter_evening_4k
    // hansaplatz_4k
    _ibl.cubemap = generate_cubemap_from_hdr("..\\..\\assets\\moonlit_golf_4k.hdr", 1024, true); 

    init_irradiance_cubemap_pipeline();
    _ibl.irradiancemap = generate_irradiance_map_from_cubemap(1024, true);

    init_prefiltered_cubemap_pipeline();
    _ibl.prefilteredmap = generate_prefiltered_map_from_cubemap(1024);

    init_brdf_integration_pipeline();
    generate_brdf_lut(256, false);

    init_ibl_descriptor_set();

    init_shadow_mapping_pipeline();
    init_shadow_mapping_descriptor_set();

    init_point_light_descriptor_set();

    update_cluster_size(120, 24);
    init_cluster_building_compute_pipeline();

    init_pipelines();

    init_imgui();
    init_default_data();

    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(-14.5, 2.5, -0.5);
    mainCamera.pitch = 0.0;
    mainCamera.yaw = 1.5;

    addRandomPointLightsInRect(512, glm::vec2(-20.0f, -20.0f), glm::vec2(20.0f, 20.0f));

    std::string helmetPath = { "..\\..\\assets\\DamagedHelmet.glb" };
    auto helmetFile = loadGltf(this, helmetPath);
    assert(helmetFile.has_value());
    loadedScenes["DamagedHelmet"] = *helmetFile;

    //std::string bistroPath = { "..\\..\\assets\\shadow.glb" };
    std::string bistroPath = { "..\\..\\assets\\Bistro.glb" };
    //std::string bistroPath = { "..\\..\\assets\\Sponza\\Sponza.gltf" };
    auto bistroFile = loadGltf(this, bistroPath);
    assert(bistroFile.has_value());
    loadedScenes["Bistro"] = *bistroFile;

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(_device);

        loadedScenes.clear();

        for (int i = 0; i < FRAME_OVERLAP; i++) {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
            _frames[i]._deletionQueue.flush();
        }

        for (int i = 0; i < _renderFinishedSemaphores.size(); i++) {
            vkDestroySemaphore(_device, _renderFinishedSemaphores[i], nullptr);
        }

        for (auto& mesh : testMeshes) {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }

        metalRoughMaterial.clear_resources(_device);

        _mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    loadedEngine = nullptr;
}

// Called before drawing each frame
void VulkanEngine::update_scene() {
    auto start = std::chrono::system_clock::now();

    //loadedScenes["DamagedHelmet"]->Draw(glm::mat4{ 1.f }, mainDrawContext);
    loadedScenes["Bistro"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    mainCamera.update(deltaTime);
    sceneData.view = mainCamera.getViewMatrix();
    nearPlane = 0.1;
    farPlane = 100.0;
    sceneData.proj = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, farPlane, nearPlane);
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(lightColor, lightIntensity);

    float s = std::sin(mZenith);
    glm::vec3 sunDir = glm::normalize(glm::vec3(
        s * std::cos(mAzimuth),
        std::cos(mZenith),
        s * std::sin(mAzimuth)
    ));
    sceneData.sunlightDirection = glm::vec4(sunDir, 0.0f);

    float lambda = 0.7f;
    _shadowRes.cascadePlanes = buildCascadePlanes(_shadowRes.numCascades, nearPlane, farPlane, lambda);
    std::vector<LightInfo> cascades = getLightSpaceMatrices(_shadowRes.numCascades, _shadowRes.cascadePlanes, nearPlane, farPlane, sceneData.view, sceneData.sunlightDirection);
    std::vector<GPUShadowMapData> cascadeData{};
    std::vector<glm::vec2> orthoDims{};
    for (int i = 0; i < cascades.size(); ++i) {
        GPUShadowMapData m{ 
            .lightSpaceMatrix = cascades[i].lightViewProj 
        };
        cascadeData.push_back(m);

        orthoDims.push_back(cascades[i].orthoDims);
    }
    _shadowRes.dataPerCascade = cascadeData;    
    _shadowRes.orthoDims = orthoDims;

    _lightRes.lightParams.lightCount = _lightRes.lights.size();
    _lightRes.builderIn.invProj = glm::inverse(sceneData.proj);
    _lightRes.builderIn.screenDimensions = glm::uvec4(_windowExtent.width, _windowExtent.height, 0, 0);
    update_cluster_size(120, 24);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.scene_update_time = elapsed.count() / 1000.f;
}

// draws into draw img (not swapchain img) in color attachment optimal layout
void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {
    stats.drawcall_count = 0;
    stats.triangle_count = 0;
    auto start = std::chrono::system_clock::now();

    // indices into `mainDrawContext.OpaqueSurfaces` vector
    // Why? We want to sort draw calls in order to minimize state changes
    // But sorting the IRenederables directly would be slow
    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(mainDrawContext.OpaqueSurfaces.size());
    for (int i = 0; i < mainDrawContext.OpaqueSurfaces.size(); i++) {
        if (is_visible(mainDrawContext.OpaqueSurfaces[i], sceneData.viewproj)) { // TODO: This frustum cull like actually sucks
            opaque_draws.push_back(i);
        }
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = mainDrawContext.OpaqueSurfaces[iA];
        const RenderObject& B = mainDrawContext.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            return A.indexBuffer < B.indexBuffer; // vulkan handle comparison
        }
        else {
            return A.material < B.material; // ptr comparison
        }
    });
    // TODO: "Another way of doing this is that we would calculate a sort key, 
    // and then our opaque_draws would be something like 20 bits draw index, and 44 bits for sort key/hash.
    // That way would be faster than this as it can be sorted through faster methods"

    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    VkViewport shadowViewport = {};
    shadowViewport.x = 0;
    shadowViewport.y = 0;
    shadowViewport.width = 1024;
    shadowViewport.height = 1024;
    shadowViewport.minDepth = 0.f;
    shadowViewport.maxDepth = 1.f;

    VkViewport viewport = {}; 
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    VkRect2D shadowScissor = {};
    shadowScissor.offset.x = 0;
    shadowScissor.offset.y = 0;
    shadowScissor.extent.width = 1024;
    shadowScissor.extent.height = 1024;

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;

    // --- Cluster Building Compute Pipeline ---

    // 1. Build cluster grid
    // TODO: Only run this if frustum changes shape, not every frame
    AllocatedBuffer froxelAABBs = build_cluster_grid();

    // 2. Z pre-pass

    // 3. Find visible clusters

    // 4. Deduplicate clusters

    // 5. Light binning (the actual clustering part)

    // --- CSM Depth Pre-Pass ---

    auto drawShadowPrepass = [&](const RenderObject& r) {
        // rebind index buffer if needed. != operator here is just comparing handles
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        // calculate final mesh matrix
        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;
        vkCmdPushConstants(cmd, _shadowRes.prepassPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);

        //stats
        stats.drawcall_count++;
        stats.triangle_count += r.indexCount / 3;
    };

    vkutil::transition_image(cmd, _shadowRes.shadowMap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    for (uint32_t c = 0; c < _shadowRes.numCascades; ++c) {
        // Begin rendering to layer c
        VkRenderingAttachmentInfo depthAtt = vkinit::depth_attachment_info(_shadowRes.layerViews[c], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri = vkinit::rendering_info(VkExtent2D{ 1024, 1024 }, nullptr, &depthAtt);
        ri.layerCount = 1; // we're using a 2D view anyway
        vkCmdBeginRendering(cmd, &ri);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowRes.prepassPipeline);

        vkCmdSetViewport(cmd, 0, 1, &shadowViewport);
        vkCmdSetScissor(cmd, 0, 1, &shadowScissor);

        // Update the prepass UBO for this cascade (light VP for cascade c)
        AllocatedBuffer shadowMapUniforms = create_buffer(sizeof(GPUShadowMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        GPUShadowMapData* u = (GPUShadowMapData*)shadowMapUniforms.allocation->GetMappedData();
        *u = _shadowRes.dataPerCascade[c]; // light VP for this cascade

        VkDescriptorSet shadowDescriptor = _shadowRes.descriptorAllocator.allocate(_device, _shadowRes.prepassSetLayout);
        DescriptorWriter shadowWriter;
        shadowWriter.write_buffer(0, shadowMapUniforms.buffer, sizeof(GPUShadowMapData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        shadowWriter.update_set(_device, shadowDescriptor);

        get_current_frame()._deletionQueue.push_function([=, this]() {
            destroy_buffer(shadowMapUniforms);
        });

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowRes.prepassPipelineLayout, 0, 1, &shadowDescriptor, 0, nullptr);

        // depth bias per cascade
        //vkCmdSetDepthBias(cmd, constBias, slopeBias, clamp);

        for (auto& r : mainDrawContext.OpaqueSurfaces) {
            drawShadowPrepass(r);
        }

        vkCmdEndRendering(cmd);
    }

    vkutil::transition_image(cmd, _shadowRes.shadowMap.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkClearValue clearValue = { 0.0, 0.0, 0.0, 0.0 };
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, &clearValue, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); // begin a render pass connected to our draw image
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // --- Actual Render (Skybox Pipeline + PBR Pipeline) ---

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipeline);

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // scene data (mvp, dir light, ambient light)
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, globalDescriptor);
    }
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
    });

    {
        VkDescriptorSet skyboxSets[] = {
            globalDescriptor, 
            _ibl.iblSet 
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipelineLayout, 0, 2, skyboxSets, 0, nullptr);

        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // Cascades data
    AllocatedBuffer gpuCascadesBuffer = create_buffer(sizeof(GPUShadowCascades), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto* csmUBO = reinterpret_cast<GPUShadowCascades*>(gpuCascadesBuffer.allocation->GetMappedData());
    for (uint32_t i = 0; i < NUM_CASCADES; ++i) {
        csmUBO->lightViewProj[i] = _shadowRes.dataPerCascade[i].lightSpaceMatrix;
    }
    for (uint32_t i = 0; i <= NUM_CASCADES; ++i) {
        csmUBO->splitDepths[i].v = _shadowRes.cascadePlanes[i]; // view-space distances
    }
    for (uint32_t i = 0; i < NUM_CASCADES; ++i) {
        csmUBO->orthoDims[i].v = _shadowRes.orthoDims[i];
    }
    _shadowRes.shadowUboSet = _shadowRes.descriptorAllocator.allocate(_device, _shadowRes.shadowUboSetLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuCascadesBuffer.buffer, sizeof(GPUShadowCascades), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, _shadowRes.shadowUboSet);
    }
    get_current_frame()._deletionQueue.push_function([=, this] {
        destroy_buffer(gpuCascadesBuffer);
    });

    // Point light data
    size_t pointLightsSize = _lightRes.lights.size() * sizeof(PointLight);
    AllocatedBuffer pointLightBuffer = create_buffer(pointLightsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // TODO: Unsure if below will work, different from before
    void* pointLightsPtr = pointLightBuffer.info.pMappedData; 
    std::memcpy(pointLightsPtr, _lightRes.lights.data(), pointLightsSize);
    AllocatedBuffer lightParamBuffer = create_buffer(sizeof(LightParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto* lightParamPtr = reinterpret_cast<LightParams*>(lightParamBuffer.allocation->GetMappedData());
    *lightParamPtr = _lightRes.lightParams;
    // TODO: Alloc descriptor set on the fly here? It's what I did before
    _lightRes.set = _lightRes.descriptorAllocator.allocate(_device, _lightRes.setLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, pointLightBuffer.buffer, pointLightsSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(3, lightParamBuffer.buffer, sizeof(LightParams), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, _lightRes.set);
    }
    get_current_frame()._deletionQueue.push_function([=, this] {
        destroy_buffer(pointLightBuffer);
        destroy_buffer(lightParamBuffer);
    });

    // state tracking, used to avoid redundant re-bindings in consecutive calls to `draw` lambda
    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& r) {
        if (r.material != lastMaterial) { // rebind pipeline and descriptors if the material changed
            lastMaterial = r.material;
            if (r.material->pipeline != lastPipeline) {
                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 2, 1, &_ibl.iblSet, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 3, 1, &_shadowRes.shadowSet, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 4, 1, &_shadowRes.shadowUboSet, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 5, 1, &_lightRes.set, 0, nullptr);

                VkViewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = (float)_drawExtent.width;
                viewport.height = (float)_drawExtent.height;
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _drawExtent.width;
                scissor.extent.height = _drawExtent.height;
                vkCmdSetScissor(cmd, 0, 1, &scissor);
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1, &r.material->materialSet, 0, nullptr);
        }

        // rebind index buffer if needed. != operator here is just comparing handles
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        // calculate final mesh matrix
        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;
        vkCmdPushConstants(cmd, r.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);

        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);

        //stats
        stats.drawcall_count++;
        stats.triangle_count += r.indexCount / 3;
    };

    for (auto& r : opaque_draws) { // indices into OpaqueSurfaces sorted in order to minimize state changes
        draw(mainDrawContext.OpaqueSurfaces[r]);
    }

    for (auto& r : mainDrawContext.TransparentSurfaces) {
        draw(r);
    }

    vkCmdEndRendering(cmd);

    // we delete the draw commands now that we processed them
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.f;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {    
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect]; // idx edited by ImGui
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline); // compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr); // bind draw img for comp shaders
    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data); // pc data edited by ImGui
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw() {
    update_scene();

    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resizeRequested = true;
        return; // early return if swapchain img invalidated (i.e. due to window resize)
    }

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer; // copy, host cmd buffer is really just 64 bit ptr

    VK_CHECK(vkResetCommandBuffer(cmd, 0)); // safe after fence
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT); // 1 submit per frame before the command buffer is reset

    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); // gfx pipeline draws into color optimal layout
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    draw_geometry(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); // copy draw img -> swapchain img
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); // for imgui
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR); // present layout

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore); // gpu waits on swapchain img acquisition

    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, _renderFinishedSemaphores[swapchainImageIndex]); // signal the render semaphore once the work we are about to submit finishes; presentation will wait on this semaphore

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence)); // we'll block on this submit fence when waiting on it at the top of draw

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;
    presentInfo.pWaitSemaphores = &_renderFinishedSemaphores[swapchainImageIndex]; // gpu waits on submitted work to finish before presenting
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resizeRequested = true;
    }

    _frameNumber++;
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    Uint64 NOW = SDL_GetPerformanceCounter();
    Uint64 LAST = 0;
    deltaTime = 0;

    // main loop
    while (!bQuit) {
        LAST = NOW;
        NOW = SDL_GetPerformanceCounter();
        deltaTime = (double)((NOW - LAST) * 1000 / (double)SDL_GetPerformanceFrequency());

        // begin clock
        auto start = std::chrono::system_clock::now();

        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                bQuit = true;
            }
            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL2_ProcessEvent(&e); // send event to imgui
        }

        if (stop_rendering) { // do not draw if we are minimized
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // throttled
            continue;
        }

        if (resizeRequested) {
            resize_swapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();

        ImGui::NewFrame();

        ImGui::Begin("Stats");
        ImGui::Text("frametime %f ms", stats.frametime);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time);
        ImGui::Text("update time %f ms", stats.scene_update_time);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::End();

        // Lighting controls
        ImGui::Begin("Directional Light");

        // Sliders shown in degrees, values kept in radians
        ImGui::SliderAngle("Azimuth", &mAzimuth, -180.0f, 180.0f);
        ImGui::SliderAngle("Zenith", &mZenith, 0.0f, 180.0f);
        ImGui::SliderFloat("Distance", &lightDist, 1.0f, 10000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Intensity", &lightIntensity, 1.0f, 10000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::ColorEdit3("Light Color", &lightColor[0]);

        // Spherical (azimuth around Y, zenith from +Y) -> cartesian
        // y = cos(zenith), radius = sin(zenith)
        // x = radius * cos(azimuth), z = radius * sin(azimuth)
        float s = std::sin(mZenith);
        glm::vec3 sunDir = glm::normalize(glm::vec3(
            s * std::cos(mAzimuth),   // x
            std::cos(mZenith),        // y
            s * std::sin(mAzimuth)    // z
        ));

        ImGui::Text("dir = [%.3f, %.3f, %.3f]", sunDir.x, sunDir.y, sunDir.z);
        ImGui::End();

        ImGui::Begin("Camera");
        ImGui::Text("Position: X = %.2f, Y = %.2f, Z = %.2f",
            mainCamera.position.x,
            mainCamera.position.y,
            mainCamera.position.z);
        ImGui::Text("Pitch: %.2f", mainCamera.pitch);
        ImGui::Text("Yaw:   %.2f", mainCamera.yaw);
        ImGui::End();

        ImGui::Begin("Point Lights");

        // Add button — pushes a default light
        if (ImGui::Button("Add point light")) {
            PointLight pl{};
            pl.color_intensity = glm::vec4(1.f, 1.f, 1.f, 1.f); // color (rgb) + intensity
            pl.pos_radius = glm::vec4(0.f, 0.f, 0.f, 1.f); // position (xyz) + radius
            _lightRes.lights.push_back(pl);
        }

        // Small status line
        ImGui::SameLine();
        ImGui::Text("Count: %zu", _lightRes.lights.size());

        ImGui::Separator();

        // Collapsible sections for existing lights (with per-light remove)
        for (size_t i = 0; i < _lightRes.lights.size(); /* incremented manually */) {
            ImGui::PushID(static_cast<int>(i));
            auto& pl = _lightRes.lights[i];

            // Each light gets its own collapsible header
            const std::string label = "Light " + std::to_string(i);
            if (ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {

                // Position (xyz) + Radius (w)
                ImGui::DragFloat3("Position", &pl.pos_radius.x, 0.1f);
                ImGui::DragFloat("Radius", &pl.pos_radius.w, 0.05f, 0.0f, 1e6f);
                if (pl.pos_radius.w < 0.0f) pl.pos_radius.w = 0.0f;

                // Color (rgb) + Intensity (w)
                ImGui::ColorEdit3("Color", &pl.color_intensity.x);
                ImGui::DragFloat("Intensity", &pl.color_intensity.w, 0.05f, 0.0f, 1e6f);
                if (pl.color_intensity.w < 0.0f) pl.color_intensity.w = 0.0f;

                // Remove button for this light
                if (ImGui::Button("Remove")) {
                    _lightRes.lights.erase(_lightRes.lights.begin() + static_cast<long>(i));
                    ImGui::PopID();
                    // do not increment i; the next element shifts into index i
                    continue;
                }
            }

            ImGui::PopID();
            ++i;
        }

        // Keep the GPU-facing count in sync
        _lightRes.lightParams.lightCount = static_cast<uint32_t>(_lightRes.lights.size());

        ImGui::End();

        ImGui::Render();

        draw();

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
    }
}

// GLTF Metal-Roughness functions

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine) {
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh_pbr.frag.spv", engine->_device, &meshFragShader)) {
        fmt::println("Error when building the triangle fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh_pbr.vert.spv", engine->_device, &meshVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // Material Constants UBO
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Base Color
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Metalness+Roughness
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Normal
    layoutBuilder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // AO
    layoutBuilder.add_binding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Emissive

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { 
        engine->_gpuSceneDataDescriptorLayout, 
        materialLayout,
        engine->_ibl.iblSetLayout, 
        engine->_shadowRes.shadowSetLayout, 
        engine->_shadowRes.shadowUboSetLayout,
        engine->_lightRes.setLayout
    };
    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 6;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // These are the only differences for the transparent pipeline
    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

void GLTFMetallic_Roughness::clear_resources(VkDevice device) {
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
    vkDestroyPipelineLayout(device, transparentPipeline.layout, nullptr); // opaque pipeline uses a handle to the same pipeline layout
    vkDestroyPipeline(device, transparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, opaquePipeline.pipeline, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator) {
    MaterialInstance matData;

    // select opaque or transparent pipeline
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    }
    else {
        matData.pipeline = &opaquePipeline;
    }

    // alloc. descriptor set
    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    // write material constants (ubo) and material maps (textures+samplers)
    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // Material Constants
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, // base color
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, // metalness + roughness
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(3, resources.normalImage.imageView, resources.normalSampler, // normal
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(4, resources.occlusionImage.imageView, resources.occlusionSampler, // ao
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(5, resources.emissiveImage.imageView, resources.emissiveSampler, // emissive
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    // return as struct to be used during command buffer recording
    return matData;
}

// Mesh Node functions

void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx) {
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;
        def.bounds = s.bounds;

        if (s.material->data.passType == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        }
        else {
            ctx.OpaqueSurfaces.push_back(def);
        }
        
    }

    // recurse down
    Node::Draw(topMatrix, ctx);
}

// Clustered Shading

AllocatedBuffer VulkanEngine::build_cluster_grid() {
    size_t outBufSize = _lightRes.numClusters * sizeof(ClusterBuilderOut);
    AllocatedBuffer outBuf = create_buffer(outBufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY); // this will only be used by other shaders

    size_t inBufSize = sizeof(ClusterBuilderIn);
    AllocatedBuffer inBuf = create_buffer(inBufSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto* inBufPtr = reinterpret_cast<ClusterBuilderIn*>(inBuf.allocation->GetMappedData());
    *inBufPtr = _lightRes.builderIn;
    //vmaFlushAllocation(allocator, inBuf.allocation, 0, inBufSize);

    _lightRes.builderSet = _lightRes.descriptorAllocator.allocate(_device, _lightRes.builderSetLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, outBuf.buffer, outBufSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, inBuf.buffer, inBufSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, _lightRes.builderSet);
    }
    get_current_frame()._deletionQueue.push_function([=, this] {
        destroy_buffer(outBuf);
        destroy_buffer(inBuf);
    });

    immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _lightRes.builderPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _lightRes.builderPipelineLayout, 0, 1, &_lightRes.builderSet, 0, nullptr);

        ClusterBuilderPushConstantsIn pc;
        pc.nearPlane = nearPlane;
        pc.farPlane = farPlane;
        vkCmdPushConstants(cmd, _lightRes.builderPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterBuilderPushConstantsIn), &pc);

        vkCmdDispatch(cmd, _lightRes.tiles.x, _lightRes.tiles.y, _lightRes.tiles.z);
    });

    return outBuf;
}

void VulkanEngine::update_cluster_size(uint32_t tileSizePx, uint32_t numZSlices) {
    _lightRes.builderIn.tileSizes[3] = tileSizePx;
    _lightRes.tiles.x = (_windowExtent.width + tileSizePx - 1) / tileSizePx;
    _lightRes.tiles.y = (_windowExtent.height + tileSizePx - 1) / tileSizePx;
    _lightRes.tiles.z = numZSlices;
    _lightRes.numClusters = _lightRes.tiles.x * _lightRes.tiles.y * _lightRes.tiles.z;
}

void VulkanEngine::init_cluster_building_compute_pipeline() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // array of froxel AABBs (output)
    b.add_binding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // inv proj, tile size, screen dims (input)
    _lightRes.builderSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(ClusterBuilderPushConstantsIn); // 
    pc.offset = 0;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &_lightRes.builderSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(_device, &plci, nullptr, &_lightRes.builderPipelineLayout));

    // shader & pipeline
    VkShaderModule cs;
    if (!(vkutil::load_shader_module("../../shaders/cluster_builder.comp.spv", _device, &cs))) {
        fmt::print("Error when building the cluster builder compute shader \n");
    }
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.layout = _lightRes.builderPipelineLayout;
    cpci.stage = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cs,
        .pName = "main"
    };
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_lightRes.builderPipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
}

// Point lights

void VulkanEngine::init_point_light_descriptor_set() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // lights
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // offsets
    b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // indices
    b.add_binding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // params
    _lightRes.setLayout = b.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);

    DescriptorAllocatorGrowable::PoolSizeRatio sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8.0f },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8.0f }
    };
    _lightRes.descriptorAllocator = DescriptorAllocatorGrowable{};
    _lightRes.descriptorAllocator.init(_device, 64, std::span(sizes));
    _lightRes.set = _lightRes.descriptorAllocator.allocate(_device, _lightRes.setLayout);
}

// Shadow mapping

void VulkanEngine::init_shadow_mapping_pipeline() {
    VkShaderModule shadowFragShader;
    if (!vkutil::load_shader_module("../../shaders/shadow_map.frag.spv", _device, &shadowFragShader)) {
        fmt::println("Error when building the shadow mapping fragment shader module");
    }

    VkShaderModule shadowVertexShader;
    if (!vkutil::load_shader_module("../../shaders/shadow_map.vert.spv", _device, &shadowVertexShader)) {
        fmt::println("Error when building the shadow mapping vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants); // model matrix, vertex buffer device address (i.e. vertex pulling)
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    _shadowRes.prepassSetLayout = b.build(_device, VK_SHADER_STAGE_VERTEX_BIT);

    {
        DescriptorAllocatorGrowable::PoolSizeRatio sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8.0f },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8.0f }
        };
        _shadowRes.descriptorAllocator = DescriptorAllocatorGrowable{};
        _shadowRes.descriptorAllocator.init(_device, 64, std::span(sizes));
        _shadowRes.prepassSet = _shadowRes.descriptorAllocator.allocate(_device, _shadowRes.prepassSetLayout);
    }

    VkDescriptorSetLayout layouts[] = { _shadowRes.prepassSetLayout };
    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 1;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_layout_info, nullptr, &newLayout));

    _shadowRes.prepassPipelineLayout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(shadowVertexShader, shadowFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    // render format
    _shadowRes.numCascades = NUM_CASCADES;
    _shadowRes.shadowMap = create_image(
        VkExtent3D{ 1024, 1024, 1 },
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        false,
        NUM_CASCADES,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY
    );

    // Create individual image views into each layer so we can render into them during shadow prepass
    // .shadowMap.imageView is a view into a 2d array, i.e. a sampler2DArray. we can't use that as a rendering attachment
    auto aspect = VK_IMAGE_ASPECT_DEPTH_BIT; 
    _shadowRes.layerViews.resize(NUM_CASCADES);
    for (uint32_t i = 0; i < NUM_CASCADES; ++i) {
        _shadowRes.layerViews[i] = create_view(_shadowRes.shadowMap.image, _shadowRes.shadowMap.imageFormat, aspect, VK_IMAGE_VIEW_TYPE_2D, 0, 1, i, 1);
    }

    pipelineBuilder.set_depth_format(_shadowRes.shadowMap.imageFormat);
    pipelineBuilder.disable_color_attachments();

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    _shadowRes.prepassPipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, shadowFragShader, nullptr);
    vkDestroyShaderModule(_device, shadowVertexShader, nullptr);
}

void VulkanEngine::init_shadow_mapping_descriptor_set() {
    {
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // shadow map
        _shadowRes.shadowSetLayout = b.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        _shadowRes.shadowSet = _shadowRes.descriptorAllocator.allocate(_device, _shadowRes.shadowSetLayout);

        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter = VK_FILTER_LINEAR; // PCF requires linear filter
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        si.compareEnable = VK_FALSE;
        si.compareOp = VK_COMPARE_OP_ALWAYS; // check this
        si.minLod = 0.0f;
        si.maxLod = 0.0f;
        vkCreateSampler(_device, &si, nullptr, &_shadowRes.shadowSampler);

        DescriptorWriter w;
        w.write_image(0, _shadowRes.shadowMap.imageView, _shadowRes.shadowSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        w.update_set(_device, _shadowRes.shadowSet);
    }

    {
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); 
        _shadowRes.shadowUboSetLayout = b.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }
}

LightInfo VulkanEngine::compute_light_space_matrix(float near, float far, const glm::mat4& view, const glm::vec4& lightDir4) {
    constexpr float fovY = glm::radians(70.f);
    const float aspect = (float)_windowExtent.width / (float)_windowExtent.height;

    // cascade slice frustum corners
    auto cornersWS = frustumCornersWS_fromView(view, fovY, aspect, near, far);

    // frustum slice center
    glm::vec3 center(0.0f);
    for (auto& c : cornersWS) {
        center += c;
    }
    center /= 8.0f;

    // shadow map render eye
    glm::vec3 lightDir = glm::normalize(glm::vec3(lightDir4));
    glm::vec3 lightPos = center - lightDir;

    glm::mat4 lightView = glm::lookAtRH(lightPos, center, glm::vec3(0, 1, 0));

    // fit ortho bounds to frustum slice
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;
    for (auto& c : cornersWS) {
        glm::vec4 lc = lightView * glm::vec4(c, 1.0f);
        minX = std::min(minX, lc.x); 
        maxX = std::max(maxX, lc.x);
        minY = std::min(minY, lc.y); 
        maxY = std::max(maxY, lc.y);
        minZ = std::min(minZ, lc.z); 
        maxZ = std::max(maxZ, lc.z);
    }

    // stabilize to texel size to reduce shimmering
    const float mapRes = float(1024); // TODO: variable
    float worldUnitsPerTexelX = (maxX - minX) / mapRes;
    float worldUnitsPerTexelY = (maxY - minY) / mapRes;
    auto snap = [](float v, float s) { 
        return std::floor(v / s) * s; 
    };
    minX = snap(minX, worldUnitsPerTexelX); 
    maxX = snap(maxX, worldUnitsPerTexelX);
    minY = snap(minY, worldUnitsPerTexelY); 
    maxY = snap(maxY, worldUnitsPerTexelY);

    // pad so shadows don't disappear due to objects falling outside ortho bounds
    const float zPad = 25.0f; // tunable
    minZ -= zPad;
    maxZ += zPad;

    // Vulkan ortho
    glm::mat4 lightProj = glm::orthoRH_ZO(minX, maxX, minY, maxY, -maxZ, -minZ);
    float orthoWidth = maxX - minX;
    float orthoHeight = maxY - minY;

    LightInfo ret = {
        .lightViewProj = lightProj * lightView,
        .orthoDims = glm::vec2(orthoWidth, orthoHeight)
    };

    return ret;
}

std::vector<LightInfo> VulkanEngine::getLightSpaceMatrices(uint32_t num_cascades, const std::vector<float>& cascade_planes, float near_plane, float far_plane, const glm::mat4& view, const glm::vec4& lightDir) {
    assert(cascade_planes.size() == num_cascades + 1);
    std::vector<LightInfo> ret;
    ret.reserve(num_cascades);

    for (uint32_t i = 0; i < num_cascades; ++i) {
        float n = cascade_planes[i];
        float f = cascade_planes[i + 1];
        LightInfo temp = compute_light_space_matrix(n, f, view, lightDir);
        ret.emplace_back(temp);
    }
    return ret;
}

// IBL

void VulkanEngine::init_brdf_integration_pipeline() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // output prefiltered cubemap
    _ibl.brdfSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(IrradiancePushConstants); // Placeholder, not sure what/if I need for PCs
    pc.offset = 0;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &_ibl.brdfSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(_device, &plci, nullptr, &_ibl.brdfPipelineLayout));

    // shader & pipeline
    VkShaderModule cs;
    if (!(vkutil::load_shader_module("../../shaders/brdflut.comp.spv", _device, &cs))) {
        fmt::print("Error when building the BRDF LUT compute shader \n");
    }
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.layout = _ibl.brdfPipelineLayout;
    cpci.stage = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cs,
        .pName = "main"
    };
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_ibl.brdfPipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
}

AllocatedImage VulkanEngine::generate_brdf_lut(uint32_t size, bool mipmapped) {
    _ibl.brdfLUT = create_image( // compute shader will write to this, create_image just allocates memory and creates VkImage + VkImageView. No actual data yet
        VkExtent3D{ size, size, 1 }, 
        VK_FORMAT_R16G16B16A16_SFLOAT, // no idea what BRDF LUTs are usually stored as
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, // not sure, guessing here
        mipmapped // not sure if I do actually need mipmaps for this?
    );

    VkImageViewCreateInfo arr = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _ibl.brdfLUT.image, VK_IMAGE_ASPECT_COLOR_BIT);
    arr.viewType = VK_IMAGE_VIEW_TYPE_2D;
    arr.subresourceRange.baseMipLevel = 0;
    arr.subresourceRange.levelCount = 1; // writing only mip 0 here
    arr.subresourceRange.baseArrayLayer = 0;
    arr.subresourceRange.layerCount = 1;
    VkImageView lutView{};
    VK_CHECK(vkCreateImageView(_device, &arr, nullptr, &lutView));

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _ibl.brdfLUT.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    _ibl.brdfSet = _ibl.descriptorAllocator.allocate(_device, _ibl.brdfSetLayout);

    {
        DescriptorWriter w;
        w.write_image(0, lutView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(_device, _ibl.brdfSet);
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.brdfPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.brdfPipelineLayout, 0, 1, &_ibl.brdfSet, 0, nullptr);

        const uint32_t group = 8;
        const uint32_t gx = (size + group - 1) / group;
        const uint32_t gy = (size + group - 1) / group;

        IrradiancePushConstants pc;
        pc.size = size;
        pc.sampleCount = 64;
        vkCmdPushConstants(cmd, _ibl.brdfPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IrradiancePushConstants), &pc);

        vkCmdDispatch(cmd, gx, gy, 1);

        vkutil::transition_image(cmd, _ibl.brdfLUT.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    });

    if (mipmapped) {
        immediate_submit([&](VkCommandBuffer cmd) {
            vkutil::generate_mipmaps(cmd, _ibl.brdfLUT.image, VkExtent2D{ _ibl.brdfLUT.imageExtent.width, _ibl.brdfLUT.imageExtent.height });
        });
    }
    else {
        immediate_submit([&](VkCommandBuffer cmd) {
            vkutil::transition_image(cmd, _ibl.brdfLUT.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    }

    // cleanup the temporary per-face views
    vkDestroyImageView(_device, lutView, nullptr);

    return _ibl.brdfLUT;
}

void VulkanEngine::init_prefiltered_cubemap_pipeline() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // input cubemap
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_MIPS); // output prefiltered cubemap
    _ibl.prefilterSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(PrefilterPushConstants); // size, sample count, mip level, roughness
    pc.offset = 0;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &_ibl.prefilterSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(_device, &plci, nullptr, &_ibl.prefilterPipelineLayout));

    // shader & pipeline
    VkShaderModule cs;
    if (!(vkutil::load_shader_module("../../shaders/prefilter.comp.spv", _device, &cs))) {
        fmt::print("Error when building the prefiltering compute shader \n");
    }
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.layout = _ibl.prefilterPipelineLayout;
    cpci.stage = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cs,
        .pName = "main"
    };
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_ibl.prefilterPipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
}

AllocatedImage VulkanEngine::generate_prefiltered_map_from_cubemap(uint32_t cubeSize) {
    _ibl.prefilteredmap = create_cubemap_image(cubeSize, VK_FORMAT_R16G16B16A16_SFLOAT, true);

    std::vector<VkImageView> prefilterMipViews;
    const uint32_t totalMips = (uint32_t)std::floor(std::log2(cubeSize)) + 1;
    prefilterMipViews.reserve(totalMips);
    for (uint32_t m = 0; m < totalMips; ++m) {
        VkImageViewCreateInfo v = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _ibl.prefilteredmap.image, VK_IMAGE_ASPECT_COLOR_BIT);
        v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.subresourceRange.baseMipLevel = m;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.baseArrayLayer = 0;
        v.subresourceRange.layerCount = 6;
        VkImageView view;
        VK_CHECK(vkCreateImageView(_device, &v, nullptr, &view));
        prefilterMipViews.push_back(view);
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _ibl.prefilteredmap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    });

    _ibl.prefilterSet = _ibl.descriptorAllocator.allocate(_device, _ibl.prefilterSetLayout);

    {
        DescriptorWriter w;
        w.write_image(0, _ibl.cubemap.imageView, _ibl.linearClampSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

        for (uint32_t m = 0; m < totalMips; ++m) {
            w.write_image(1, m, prefilterMipViews[m], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        }
        for (uint32_t m = totalMips; m < MAX_MIPS; ++m) {
            w.write_image(1, m, prefilterMipViews.back(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        }

        w.update_set(_device, _ibl.prefilterSet);
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.prefilterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.prefilterPipelineLayout, 0, 1, &_ibl.prefilterSet, 0, nullptr);

        const uint32_t group = 8;
        PrefilterPushConstants pc{};
        pc.sampleCount = 64;

        for (uint32_t mip = 0; mip < totalMips; ++mip) {
            const uint32_t mipSize = std::max(1u, cubeSize >> mip);
            const uint32_t gx = (mipSize + group - 1) / group;
            const uint32_t gy = (mipSize + group - 1) / group;

            pc.size = mipSize;
            pc.mipLevel = mip;
            pc.roughness = (totalMips > 1) ? float(mip) / float(totalMips - 1) : 0.0f;

            vkCmdPushConstants(cmd, _ibl.prefilterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PrefilterPushConstants), &pc);

            // z dimension = 6 faces
            vkCmdDispatch(cmd, gx, gy, 6);
            // no barrier needed: each mip is a disjoint subresource
        }

        // after all writes, make the whole image shader-readable for later sampling
        vkutil::transition_image(cmd, _ibl.prefilteredmap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    });

    // cleanup the temporary per-face views
    for (uint32_t m = 0; m < totalMips; ++m) {
        vkDestroyImageView(_device, prefilterMipViews[m], nullptr);
    }

    return _ibl.prefilteredmap;
}

void VulkanEngine::init_irradiance_cubemap_pipeline() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // input cubemap
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // output irradiance cubemap
    _ibl.irradianceSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(IrradiancePushConstants); // face size, sample delta
    pc.offset = 0;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &_ibl.irradianceSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(_device, &plci, nullptr, &_ibl.irradiancePipelineLayout));

    // shader & pipeline
    VkShaderModule cs;
    if (!(vkutil::load_shader_module("../../shaders/irradiance_gen.comp.spv", _device, &cs))) {
        fmt::print("Error when building the irradiance compute shader \n");
    }
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.layout = _ibl.irradiancePipelineLayout;
    cpci.stage = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cs,
        .pName = "main"
    };
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_ibl.irradiancePipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
}

AllocatedImage VulkanEngine::generate_irradiance_map_from_cubemap(uint32_t cubeSize, bool mipmapped) {
    _ibl.irradiancemap = create_cubemap_image(cubeSize, VK_FORMAT_R16G16B16A16_SFLOAT, mipmapped);
    VkImageViewCreateInfo arr = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _ibl.irradiancemap.image, VK_IMAGE_ASPECT_COLOR_BIT);
    arr.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    arr.subresourceRange.baseMipLevel = 0;
    arr.subresourceRange.levelCount = 1; // writing only mip 0 here
    arr.subresourceRange.baseArrayLayer = 0;
    arr.subresourceRange.layerCount = 6;
    VkImageView cubeArrayView{};
    VK_CHECK(vkCreateImageView(_device, &arr, nullptr, &cubeArrayView));

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _ibl.irradiancemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    });

    _ibl.irradianceSet = _ibl.descriptorAllocator.allocate(_device, _ibl.irradianceSetLayout);

    {
        DescriptorWriter w;
        w.write_image(0, _ibl.cubemap.imageView, _ibl.linearClampSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        w.write_image(1, cubeArrayView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(_device, _ibl.irradianceSet);
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.irradiancePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.irradiancePipelineLayout, 0, 1, &_ibl.irradianceSet, 0, nullptr);

        const uint32_t group = 8;
        const uint32_t gx = (cubeSize + group - 1) / group;
        const uint32_t gy = (cubeSize + group - 1) / group;

        IrradiancePushConstants pc;
        pc.size = cubeSize;
        pc.sampleCount = 64;
        vkCmdPushConstants(cmd, _ibl.irradiancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(IrradiancePushConstants), &pc);

        // write all 6 faces in one go
        vkCmdDispatch(cmd, gx, gy, 6);

        vkutil::transition_image(cmd, _ibl.irradiancemap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    });

    if (mipmapped) {
        immediate_submit([&](VkCommandBuffer cmd) {
            vkutil::generate_mipmaps_cube(cmd, _ibl.irradiancemap.image, VkExtent2D{ _ibl.irradiancemap.imageExtent.width, _ibl.irradiancemap.imageExtent.height }, 6, VK_FORMAT_R16G16B16A16_SFLOAT);
        });
    }

    // cleanup the temporary per-face views
    vkDestroyImageView(_device, cubeArrayView, nullptr);

    return _ibl.irradiancemap;
}

AllocatedImage VulkanEngine::create_cubemap_image(uint32_t size, VkFormat format, bool mipmapped) {
    AllocatedImage img{};
    img.imageFormat = format;
    img.imageExtent = VkExtent3D{ size, size, 1 };

    VkImageCreateInfo info = vkinit::image_create_info(format, 
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        img.imageExtent);

    info.imageType = VK_IMAGE_TYPE_2D;
    info.arrayLayers = 6;
    info.mipLevels = mipmapped ? (uint32_t)std::floor(std::log2(size)) + 1 : 1;
    info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; // This is how Vulkan knows how to use it as a samplerCube I guess

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VK_CHECK(vmaCreateImage(_allocator, &info, &allocInfo, &img.image, &img.allocation, nullptr));

    // cube view
    VkImageViewCreateInfo vinfo = vkinit::imageview_create_info(format, img.image, VK_IMAGE_ASPECT_COLOR_BIT);
    vinfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE; // And this
    vinfo.subresourceRange.levelCount = info.mipLevels;
    vinfo.subresourceRange.layerCount = 6;
    VK_CHECK(vkCreateImageView(_device, &vinfo, nullptr, &img.imageView));
    return img;
}

void VulkanEngine::init_equirect_to_cubemap_pipeline() {
    // layouts
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _ibl.convertSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &_ibl.convertSetLayout;
    plci.pushConstantRangeCount = 0;
    plci.pPushConstantRanges = nullptr;
    VK_CHECK(vkCreatePipelineLayout(_device, &plci, nullptr, &_ibl.convertPipelineLayout));

    // shader & pipeline
    VkShaderModule cs;
    if (!(vkutil::load_shader_module("../../shaders/equirect_to_cubemap.comp.spv", _device, &cs))) {
        fmt::print("Error when building the cubemap compute shader \n");
    }
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.layout = _ibl.convertPipelineLayout;
    cpci.stage = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cs,
        .pName = "main"
    };
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_ibl.convertPipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
}

AllocatedImage VulkanEngine::generate_cubemap_from_hdr(const char* hdrPath, uint32_t cubeSize, bool mipmapped) {
    _ibl.equirect = create_equirect_image_from_hdr(this, hdrPath);

    // sampler for sampling equirect in compute
    if (_ibl.linearClampSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod = 0.0f;
        si.maxLod = VK_LOD_CLAMP_NONE; // (~1000.0f)
        // or clamp to the highest actual level:
        // si.maxLod   = float(numLevels - 1);
        si.mipLodBias = 0.0f;
        si.anisotropyEnable = VK_FALSE; // not needed for cube sampling/FIS
        VK_CHECK(vkCreateSampler(_device, &si, nullptr, &_ibl.linearClampSampler));
    }

    _ibl.cubemap = create_cubemap_image(cubeSize, VK_FORMAT_R16G16B16A16_SFLOAT, mipmapped);
    VkImageViewCreateInfo arr = vkinit::imageview_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _ibl.cubemap.image, VK_IMAGE_ASPECT_COLOR_BIT);
    arr.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    arr.subresourceRange.baseMipLevel = 0;
    arr.subresourceRange.levelCount = 1; // writing only mip 0 here
    arr.subresourceRange.baseArrayLayer = 0;
    arr.subresourceRange.layerCount = 6;
    VkImageView cubeArrayView{};
    VK_CHECK(vkCreateImageView(_device, &arr, nullptr, &cubeArrayView));

    // Transition images to correct layouts
    immediate_submit([&](VkCommandBuffer cmd) {
        // equirect already transitioned to SHADER_READ_ONLY_OPTIMAL by create_image()
        vkutil::transition_image(cmd, _ibl.cubemap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
     });

    {
        DescriptorAllocatorGrowable::PoolSizeRatio sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8.0f },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8.0f },
        };
        VkDescriptorSetLayout layout = _ibl.convertSetLayout;
        _ibl.descriptorAllocator = DescriptorAllocatorGrowable{};
        _ibl.descriptorAllocator.init(_device, 64, std::span(sizes));
        _ibl.convertSet = _ibl.descriptorAllocator.allocate(_device, layout);
    }

    // Populate binding 0 once (equirect sampled)
    {
        DescriptorWriter w;
        w.write_image(0, _ibl.equirect.imageView, _ibl.linearClampSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        w.write_image(1, cubeArrayView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(_device, _ibl.convertSet);
    }

    immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.convertPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _ibl.convertPipelineLayout, 0, 1, &_ibl.convertSet, 0, nullptr);

        const uint32_t group = 8;
        const uint32_t gx = (cubeSize + group - 1) / group;
        const uint32_t gy = (cubeSize + group - 1) / group;

        // write all 6 faces in one go
        vkCmdDispatch(cmd, gx, gy, 6);

        vkutil::transition_image(cmd, _ibl.cubemap.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    });

    if (mipmapped) {
        immediate_submit([&](VkCommandBuffer cmd) {
            vkutil::generate_mipmaps_cube(cmd, _ibl.cubemap.image, VkExtent2D{ _ibl.cubemap.imageExtent.width, _ibl.cubemap.imageExtent.height }, 6, VK_FORMAT_R16G16B16A16_SFLOAT);
        });
    }

    // cleanup the temporary per-face views
    vkDestroyImageView(_device, cubeArrayView, nullptr);

    return _ibl.cubemap;
}

void VulkanEngine::init_ibl_descriptor_set() {
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // cubemap
    b.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // irradiance cubemap
    b.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // prefiltered cubemap
    b.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // BRDF LUT
    _ibl.iblSetLayout = b.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);

    _ibl.iblSet = _ibl.descriptorAllocator.allocate(_device, _ibl.iblSetLayout);

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler envSampler = _ibl.linearClampSampler; 

    DescriptorWriter w;
    w.write_image(0, _ibl.cubemap.imageView, envSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    w.write_image(1, _ibl.irradiancemap.imageView, envSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    w.write_image(2, _ibl.prefilteredmap.imageView, envSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    w.write_image(3, _ibl.brdfLUT.imageView, envSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    w.update_set(_device, _ibl.iblSet);
}