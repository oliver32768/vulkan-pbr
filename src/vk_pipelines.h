#pragma once 
#include <vk_types.h>

namespace vkutil {
    bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
};

class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;

    std::vector<VkFormat> _colorAttachmentFormats;
    VkPipelineColorBlendAttachmentState _defaultBlendAttachment{
       .blendEnable = VK_FALSE,
       .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };
    std::vector<VkPipelineColorBlendAttachmentState> _blendAttachments;

    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipelineRenderingCreateInfo _renderInfo;

    PipelineBuilder() { clear(); }

    void clear();

    VkPipeline build_pipeline(VkDevice device);
    void set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
    void enable_blending_additive();
    void enable_blending_alphablend();
    void enable_depthtest(bool depthWriteEnable, VkCompareOp op);
    void disable_depthtest();
    void set_color_attachment_format(VkFormat format);
    void set_color_attachment_formats(std::span<const VkFormat> fmts);
    void set_color_attachment_formats(std::initializer_list<VkFormat> fmts);
    void disable_color_attachments();
    void set_depth_format(VkFormat format);
    void disable_blending();
    void set_multisampling_none();
    void enable_multisampling();
    void set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    void set_polygon_mode(VkPolygonMode mode);
    void set_input_topology(VkPrimitiveTopology topology);
};
