#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject

/// Version string
+ (NSString *)opencvVersion;
+ (NSDictionary *)findContours:(const float *)mask
                                width:(int)width
                               height:(int)height;

@end

NS_ASSUME_NONNULL_END
