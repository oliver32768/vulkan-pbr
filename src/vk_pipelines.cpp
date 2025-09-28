#include <vk_pipelines.h>

#include <fstream>
#include <vk_initializers.h>

void PipelineBuilder::enable_blending_additive() {
    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    att.blendEnable = VK_TRUE;
    att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    att.colorBlendOp = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att.alphaBlendOp = VK_BLEND_OP_ADD;
    
    _blendAttachments.assign(_colorAttachmentFormats.size(), att);
}

void PipelineBuilder::enable_blending_alphablend() {
    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    att.blendEnable = VK_TRUE;
    att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    att.colorBlendOp = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att.alphaBlendOp = VK_BLEND_OP_ADD;

    _blendAttachments.assign(_colorAttachmentFormats.size(), att);
}

void PipelineBuilder::enable_depthtest(bool depthWriteEnable, VkCompareOp op) {
    _depthStencil.depthTestEnable = VK_TRUE;
    _depthStencil.depthWriteEnable = depthWriteEnable;
    _depthStencil.depthCompareOp = op;
    _depthStencil.depthBoundsTestEnable = VK_FALSE;
    _depthStencil.stencilTestEnable = VK_FALSE;
    _depthStencil.front = {};
    _depthStencil.back = {};
    _depthStencil.minDepthBounds = 0.f;
    _depthStencil.maxDepthBounds = 1.f;
}

void PipelineBuilder::disable_depthtest() {
    _depthStencil.depthTestEnable = VK_FALSE;
    _depthStencil.depthWriteEnable = VK_FALSE;
    _depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
    _depthStencil.depthBoundsTestEnable = VK_FALSE;
    _depthStencil.stencilTestEnable = VK_FALSE;
    _depthStencil.front = {};
    _depthStencil.back = {};
    _depthStencil.minDepthBounds = 0.f;
    _depthStencil.maxDepthBounds = 1.f;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format) {
    _colorAttachmentFormats.clear();
    _colorAttachmentFormats.push_back(format);

    _renderInfo.colorAttachmentCount = 1;
    _renderInfo.pColorAttachmentFormats = _colorAttachmentFormats.data();

    // Keep blend array in sync
    _blendAttachments.assign(1, _defaultBlendAttachment);
}

void PipelineBuilder::set_color_attachment_formats(std::span<const VkFormat> fmts) {
    _colorAttachmentFormats.assign(fmts.begin(), fmts.end());
    _renderInfo.colorAttachmentCount = static_cast<uint32_t>(_colorAttachmentFormats.size());
    _renderInfo.pColorAttachmentFormats = _colorAttachmentFormats.data();
    _blendAttachments.assign(_colorAttachmentFormats.size(), _defaultBlendAttachment);
}

void PipelineBuilder::set_color_attachment_formats(std::initializer_list<VkFormat> fmts) {
    _colorAttachmentFormats.assign(fmts.begin(), fmts.end());
    _renderInfo.colorAttachmentCount = static_cast<uint32_t>(_colorAttachmentFormats.size());
    _renderInfo.pColorAttachmentFormats = _colorAttachmentFormats.data();
    _blendAttachments.assign(_colorAttachmentFormats.size(), _defaultBlendAttachment);
}

void PipelineBuilder::disable_color_attachments() {
    _renderInfo.colorAttachmentCount = 0;
    _renderInfo.pColorAttachmentFormats = nullptr;
}

void PipelineBuilder::set_depth_format(VkFormat format) {
    _renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::disable_blending() {
    VkPipelineColorBlendAttachmentState att = _defaultBlendAttachment;
    if (_blendAttachments.size() != _colorAttachmentFormats.size()) {
        _blendAttachments.resize(_colorAttachmentFormats.size(), att);
    }
    else {
        std::fill(_blendAttachments.begin(), _blendAttachments.end(), att);
    }
}


void PipelineBuilder::set_multisampling_none() {
    _multisampling.sampleShadingEnable = VK_FALSE; // multisampling defaulted to no multisampling (1 sample per pixel)
    _multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    _multisampling.minSampleShading = 1.0f;
    _multisampling.pSampleMask = nullptr;
    _multisampling.alphaToCoverageEnable = VK_FALSE; // no alpha to coverage either
    _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::enable_multisampling() {
    _multisampling.sampleShadingEnable = VK_FALSE; // multisampling defaulted to no multisampling (1 sample per pixel)
    _multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;
    _multisampling.minSampleShading = 1.0f;
    _multisampling.pSampleMask = nullptr;
    _multisampling.alphaToCoverageEnable = VK_FALSE; // no alpha to coverage either
    _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
    _rasterizer.cullMode = cullMode; // front, back, none
    _rasterizer.frontFace = frontFace; // cw, ccw
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode) {
    _rasterizer.polygonMode = mode;
    _rasterizer.lineWidth = 1.f;
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology) {
    _inputAssembly.topology = topology; // tris, lines, etc.
    _inputAssembly.primitiveRestartEnable = VK_FALSE; // PrimitiveRestart is used for triangle strips and line strips
}

void PipelineBuilder::set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader) {
    _shaderStages.clear();

    // Vertex -> Fragment
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
}

// Call this once pipeline state has been configured
VkPipeline PipelineBuilder::build_pipeline(VkDevice device) {
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    // Color blending for N MRTs
    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = static_cast<uint32_t>(_blendAttachments.size()),
        .pAttachments = _blendAttachments.empty() ? nullptr : _blendAttachments.data(),
    }; // pNext and logicOp no longer specified

    // completely clear VertexInputStateCreateInfo, as we have no need for it
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &_renderInfo, // dynamic rendering
        .stageCount = static_cast<uint32_t>(_shaderStages.size()),
        .pStages = _shaderStages.data(),
        .pVertexInputState = &_vertexInputInfo,
        .pInputAssemblyState = &_inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &_rasterizer,
        .pMultisampleState = &_multisampling,
        .pDepthStencilState = &_depthStencil,
        .pColorBlendState = &colorBlending,
        .layout = _pipelineLayout,
        // renderPass = VK_NULL_HANDLE with dynamic rendering
    };

    // Dynamic window resizing
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = (uint32_t)std::size(dynStates),
        .pDynamicStates = dynStates,
    };
    pipelineInfo.pDynamicState = &dynamicInfo;

    // its easy to error out on create graphics pipeline, so we handle it a bit
    // better than the common VK_CHECK case
    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        fmt::println("failed to create pipeline");
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    } else {
        return newPipeline;
    }
}

void PipelineBuilder::clear() {
    _inputAssembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    _rasterizer = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    _multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    _depthStencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    _pipelineLayout = VK_NULL_HANDLE;
    _renderInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    _shaderStages.clear();
    _colorAttachmentFormats.clear();
    _blendAttachments.clear();
    _defaultBlendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };
}

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule) {
    // open the file. With cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)file.tellg();

    // spirv expects the buffer to be uint32
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read((char*)buffer.data(), fileSize);

    // now that the file is loaded into the buffer, we can close it
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.codeSize = buffer.size() * sizeof(uint32_t); // size in bytes
    createInfo.pCode = buffer.data();

    // NOTE: Why don't we use VK_CHECK?
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

