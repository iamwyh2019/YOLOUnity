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

@end
