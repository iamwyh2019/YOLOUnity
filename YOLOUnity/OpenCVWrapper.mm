#import "OpenCVWrapper.h"

#ifdef NO
#undef NO
#endif

#import <opencv2/opencv.hpp>
#import <arm_neon.h>

@implementation OpenCVWrapper

+ (NSString *)opencvVersion {
    std::string version = CV_VERSION;
    return [NSString stringWithUTF8String:version.c_str()];
}

+ (NSArray<NSArray<NSValue *> *> *)findContours:(const float *)mask
                                         width:(int)width
                                        height:(int)height {
    // Create binary Mat with continuous memory allocation
    cv::Mat binaryMat = cv::Mat::zeros(height, width, CV_8UC1);
    uint8_t* matData = binaryMat.data;
    
    // Convert float mask to binary using NEON when available
    #if defined(__ARM_NEON)
    dispatch_apply(height, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t y) {
        const float* srcRow = mask + y * width;
        uint8_t* dstRow = matData + y * width;
        
        int x = 0;
        // Process 16 pixels at a time using NEON
        for (; x <= width - 16; x += 16) {
            float32x4_t v1 = vld1q_f32(srcRow + x);
            float32x4_t v2 = vld1q_f32(srcRow + x + 4);
            float32x4_t v3 = vld1q_f32(srcRow + x + 8);
            float32x4_t v4 = vld1q_f32(srcRow + x + 12);
            
            // Compare with 0 and convert to 8-bit
            uint32x4_t mask1 = vcgtq_f32(v1, vdupq_n_f32(0));
            uint32x4_t mask2 = vcgtq_f32(v2, vdupq_n_f32(0));
            uint32x4_t mask3 = vcgtq_f32(v3, vdupq_n_f32(0));
            uint32x4_t mask4 = vcgtq_f32(v4, vdupq_n_f32(0));
            
            // Combine results
            uint8x8_t narrow1 = vmovn_u16(vcombine_u16(vmovn_u32(mask1), vmovn_u32(mask2)));
            uint8x8_t narrow2 = vmovn_u16(vcombine_u16(vmovn_u32(mask3), vmovn_u32(mask4)));
            uint8x16_t result = vcombine_u8(narrow1, narrow2);
            
            // Store with multiplication by 255
            uint8x16_t temp = vdupq_n_u8(255);
            vst1q_u8(dstRow + x, vandq_u8(result, temp));
        }
        
        // Handle remaining pixels
        for (; x < width; x++) {
            dstRow[x] = srcRow[x] > 0 ? 255 : 0;
        }
    });
    #else
    // Fallback for non-ARM devices
    dispatch_apply(height, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t y) {
        const float* srcRow = mask + y * width;
        uint8_t* dstRow = matData + y * width;
        
        for(int x = 0; x < width; x++) {
            dstRow[x] = srcRow[x] > 0 ? 255 : 0;
        }
    });
    #endif
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    contours.reserve(100);  // Reserve space for expected number of contours
    
    cv::findContours(binaryMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
    
    // Filter small contours
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& contour) {
                return cv::contourArea(contour) < 10;
            }
        ),
        contours.end()
    );
    
    // Convert to NSArray with pre-allocation
    NSMutableArray* result = [[NSMutableArray alloc] initWithCapacity:contours.size()];
    
    // Use parallel processing only for larger sets
    if (contours.size() > 50) {
        dispatch_apply(contours.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
            const auto& contour = contours[i];
            NSMutableArray* points = [[NSMutableArray alloc] initWithCapacity:contour.size()];
            
            @autoreleasepool {
                for(const auto& point : contour) {
                    [points addObject:@(point.x)];
                    [points addObject:@(point.y)];
                }
            }
            
            @synchronized(result) {
                [result addObject:points];
            }
        });
    } else {
        // Sequential processing for smaller sets
        for (const auto& contour : contours) {
            NSMutableArray* points = [[NSMutableArray alloc] initWithCapacity:contour.size()];
            for(const auto& point : contour) {
                [points addObject:@(point.x)];
                [points addObject:@(point.y)];
            }
            [result addObject:points];
        }
    }

    return result;
}

@end
