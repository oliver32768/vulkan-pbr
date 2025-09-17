
#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {
	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);
	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);
    void generate_mipmaps_cube(
        VkCommandBuffer cmd,
        VkImage image,
        VkExtent2D imageSize,
        uint32_t layerCount,      // pass 6 for cubemaps
        VkFormat format           // used if you add the feature check
    );
    void transition_image(VkCommandBuffer cmd, VkImage image,
        VkImageLayout oldLayout, VkImageLayout newLayout,
        VkImageAspectFlags aspect,
        uint32_t baseMip, uint32_t levelCount,
        uint32_t baseLayer, uint32_t layerCount);
};