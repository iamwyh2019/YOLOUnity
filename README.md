# YOLOUnity

iOS framework for YOLO segmentation models with Unity integration.

## Setup

### OpenCV Framework

Download the OpenCV iOS framework and add it to the project:

1. Go to [OpenCV Releases](https://opencv.org/releases/)
2. Download the iOS Pack: [opencv-4.12.0-ios-framework.zip](https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip)
3. Unzip and copy `opencv2.framework` to the `Frameworks/` folder

```bash
curl -L -o opencv.zip https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip
unzip opencv.zip
mv opencv2.framework Frameworks/
```

### Models

1. Add your `.mlpackage` model files to `YOLOUnity/Models/`
2. In Xcode, select the `.mlpackage` file → File Inspector → check **YOLOUnity** under Target Membership
3. Add a case for your model in `YOLOUnity/YOLOPredictor.swift`:

```swift
// In the init() switch statement, add your model:
case "my_model":
    return try? my_model(configuration: config).model
```

Note: Xcode auto-generates a Swift class from the `.mlpackage` filename. Hyphens become underscores (e.g., `yolo11l-seg.mlpackage` → `yolo11l_seg` class).

4. Use your model name when initializing:

```swift
InitializeYOLO(modelName: "my_model", confidenceThreshold: 0.5, iouThreshold: 0.5, scaleMethod: "scaleFit")
```
