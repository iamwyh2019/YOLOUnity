#import "OpenCVWrapper.h"

#ifdef NO
#undef NO
#endif

#import <opencv2/opencv.hpp>

@implementation OpenCVWrapper

+ (NSString *)opencvVersion
{
    // CV_VERSION is a compile-time macro like "4.5.3"
    std::string version = CV_VERSION;
    return [NSString stringWithUTF8String:version.c_str()];
}

+ (NSArray<NSArray<NSValue *> *> *)findContours:(const float *)mask
                                         width:(int)width
                                        height:(int)height {
    // Create binary Mat
    cv::Mat binaryMat(height, width, CV_8UC1);
    uint8_t* matData = binaryMat.data;
    
    // Parallel conversion to binary
    dispatch_apply(height, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t y) {
        const float* srcRow = mask + y * width;
        uint8_t* dstRow = matData + y * width;
        
        for(int x = 0; x < width; x++) {
            dstRow[x] = srcRow[x] > 0 ? 255 : 0;
        }
    });
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter small contours
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& contour) {
                return cv::contourArea(contour) < 10;
            }
        ),
        contours.end()
    );
    
    // Convert to NSArray
    NSMutableArray* result = [[NSMutableArray alloc] initWithCapacity:contours.size()];
       
    dispatch_apply(contours.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
        const auto& contour = contours[i];
        NSMutableArray* points = [[NSMutableArray alloc] initWithCapacity:contour.size() * 2];
        
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

    return result;
}

@end
