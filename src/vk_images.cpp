#include <vk_images.h>
#include <vk_initializers.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void vkutil::transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier2 imageBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imageBarrier.pNext = nullptr;

    // TODO: This is inefficient; instead use StageMasks more accurate to what you are doing
    // AllCommands stage mask on the barrier means that the barrier will stop the gpu commands completely when it arrives at the barrier.
    // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; 
    
    imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    imageBarrier.oldLayout = currentLayout;
    imageBarrier.newLayout = newLayout;

    VkImageAspectFlags aspectMask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL || currentLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange = vkinit::image_subresource_range(aspectMask);
    imageBarrier.image = image;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.pNext = nullptr;

    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void vkutil::transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 imageBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imageBarrier.pNext = nullptr;

    // TODO: This is inefficient; instead use StageMasks more accurate to what you are doing
    // AllCommands stage mask on the barrier means that the barrier will stop the gpu commands completely when it arrives at the barrier.
    // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    imageBarrier.oldLayout = currentLayout;
    imageBarrier.newLayout = newLayout;

    VkImageAspectFlags aspectMask = aspect;
    imageBarrier.subresourceRange = vkinit::image_subresource_range(aspectMask);
    imageBarrier.image = image;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.pNext = nullptr;

    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void vkutil::copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize)
{
	VkImageBlit2 blitRegion{ 
		.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2, 
		.pNext = nullptr 
	};

	blitRegion.srcOffsets[1].x = srcSize.width;
	blitRegion.srcOffsets[1].y = srcSize.height;
	blitRegion.srcOffsets[1].z = 1;

	blitRegion.dstOffsets[1].x = dstSize.width;
	blitRegion.dstOffsets[1].y = dstSize.height;
	blitRegion.dstOffsets[1].z = 1;

	blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blitRegion.srcSubresource.baseArrayLayer = 0;
	blitRegion.srcSubresource.layerCount = 1;
	blitRegion.srcSubresource.mipLevel = 0;

	blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blitRegion.dstSubresource.baseArrayLayer = 0;
	blitRegion.dstSubresource.layerCount = 1;
	blitRegion.dstSubresource.mipLevel = 0;

	VkBlitImageInfo2 blitInfo{ 
		.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2, 
		.pNext = nullptr 
	};

	blitInfo.dstImage = destination;
	blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	blitInfo.srcImage = source;
	blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	blitInfo.filter = VK_FILTER_LINEAR;
	blitInfo.regionCount = 1;
	blitInfo.pRegions = &blitRegion;

	vkCmdBlitImage2(cmd, &blitInfo);
}

void vkutil::generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize) {
    int mipLevels = int(std::floor(std::log2(std::max(imageSize.width, imageSize.height)))) + 1;
    for (int mip = 0; mip < mipLevels; mip++) {
        VkExtent2D halfSize = imageSize;
        halfSize.width /= 2;
        halfSize.height /= 2;

        VkImageMemoryBarrier2 imageBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr };

        imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange = vkinit::image_subresource_range(aspectMask);
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.baseMipLevel = mip;
        imageBarrier.image = image;

        VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .pNext = nullptr };
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &imageBarrier;

        vkCmdPipelineBarrier2(cmd, &depInfo);

        if (mip < mipLevels - 1) {
            VkImageBlit2 blitRegion{ .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2, .pNext = nullptr };

            blitRegion.srcOffsets[1].x = imageSize.width;
            blitRegion.srcOffsets[1].y = imageSize.height;
            blitRegion.srcOffsets[1].z = 1;

            blitRegion.dstOffsets[1].x = halfSize.width;
            blitRegion.dstOffsets[1].y = halfSize.height;
            blitRegion.dstOffsets[1].z = 1;

            blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.srcSubresource.baseArrayLayer = 0;
            blitRegion.srcSubresource.layerCount = 1; // was 1
            blitRegion.srcSubresource.mipLevel = mip;

            blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.dstSubresource.baseArrayLayer = 0;
            blitRegion.dstSubresource.layerCount = 1; // was 1
            blitRegion.dstSubresource.mipLevel = mip + 1;

            VkBlitImageInfo2 blitInfo{ .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2, .pNext = nullptr };
            blitInfo.dstImage = image;
            blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            blitInfo.srcImage = image;
            blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            blitInfo.filter = VK_FILTER_LINEAR;
            blitInfo.regionCount = 1;
            blitInfo.pRegions = &blitRegion;

            vkCmdBlitImage2(cmd, &blitInfo);

            imageSize = halfSize;
        }
    }

    // transition all mip levels into the final read_only layout
    transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void vkutil::generate_mipmaps_cube(
    VkCommandBuffer cmd,
    VkImage image,
    VkExtent2D imageSize,
    uint32_t layerCount,
    VkFormat format
) {
    const int mipLevels = int(std::floor(std::log2(std::max(imageSize.width, imageSize.height)))) + 1;

    // verify linear blit support
    //VkFormatProperties props; vkGetPhysicalDeviceFormatProperties(physDev, format, &props);
    //assert(props.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

    VkExtent2D srcSize = imageSize;

    for (int i = 1; i < mipLevels; i++) {
        // Make previous level (i-1) readable by blit as SRC
        {
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel = i - 1;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount = layerCount; // or VK_REMAINING_ARRAY_LAYERS

            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Blit all array layers from mip (i-1) -> mip (i)
        {
            VkImageBlit2 region{ VK_STRUCTURE_TYPE_IMAGE_BLIT_2 };
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.mipLevel = i - 1;
            region.srcSubresource.baseArrayLayer = 0;
            region.srcSubresource.layerCount = layerCount; // 6 for cubemap

            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.mipLevel = i;
            region.dstSubresource.baseArrayLayer = 0;
            region.dstSubresource.layerCount = layerCount;

            region.srcOffsets[0] = { 0, 0, 0 };
            region.srcOffsets[1] = { int32_t(srcSize.width), int32_t(srcSize.height), 1 };

            VkExtent2D dstSize = {
                std::max(1u, srcSize.width >> 1),
                std::max(1u, srcSize.height >> 1)
            };
            region.dstOffsets[0] = { 0, 0, 0 };
            region.dstOffsets[1] = { int32_t(dstSize.width), int32_t(dstSize.height), 1 };

            VkBlitImageInfo2 blit{ VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2 };
            blit.srcImage = image;
            blit.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            blit.dstImage = image;
            blit.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            blit.filter = VK_FILTER_LINEAR;
            blit.regionCount = 1;
            blit.pRegions = &region;

            vkCmdBlitImage2(cmd, &blit);

            srcSize = dstSize;
        }
        // dont immediately transition mip i-1 to SHADER_READ_ONLY here; we batch at the end
    }

    // 3) Final transitions to shader-read:
    // [0 .. mipLevels-2] : TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY
    // [mipLevels-1]      : TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY
    VkImageMemoryBarrier2 finals[2]{};
    finals[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    finals[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    finals[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    finals[0].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finals[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    finals[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    finals[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    finals[0].image = image;
    finals[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    finals[0].subresourceRange.baseMipLevel = 0;
    finals[0].subresourceRange.levelCount = std::max(0, mipLevels - 1);
    finals[0].subresourceRange.baseArrayLayer = 0;
    finals[0].subresourceRange.layerCount = layerCount;

    finals[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    finals[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    finals[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    finals[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finals[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    finals[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    finals[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    finals[1].image = image;
    finals[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    finals[1].subresourceRange.baseMipLevel = mipLevels - 1;
    finals[1].subresourceRange.levelCount = 1;
    finals[1].subresourceRange.baseArrayLayer = 0;
    finals[1].subresourceRange.layerCount = layerCount;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = (mipLevels > 1) ? 2u : 1u; // if only 1 level, only finals[1] is valid
    dep.pImageMemoryBarriers = (mipLevels > 1) ? finals : &finals[1];
    vkCmdPipelineBarrier2(cmd, &dep);
}

void vkutil::transition_image(VkCommandBuffer cmd, VkImage image,
    VkImageLayout oldLayout, VkImageLayout newLayout,
    VkImageAspectFlags aspect,
    uint32_t baseMip, uint32_t levelCount,
    uint32_t baseLayer, uint32_t layerCount)
{
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.baseMipLevel = baseMip;
    b.subresourceRange.levelCount = levelCount;      // VK_REMAINING_MIP_LEVELS ok
    b.subresourceRange.baseArrayLayer = baseLayer;
    b.subresourceRange.layerCount = layerCount;      // VK_REMAINING_ARRAY_LAYERS ok

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}