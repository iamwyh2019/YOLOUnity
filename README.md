# YOLOUnity

An iOS framework that runs YOLO instance-segmentation models with Core ML and returns detections
(class, score, box, contour polygons, centroid) through a plain C API, so it can be called from
Unity (`[DllImport("__Internal")]`) or any other native host.

It is the inference engine behind [NavSight](https://github.com/MadisonAbilityLab/NavSight), which
ships the Mapillary Vistas street-scene models (`vistas-{s,m,l}-v2`).

- Core ML through Vision, on the CPU and Neural Engine (the GPU is left free for rendering)
- Post-processing on device: confidence filtering, per-class NMS, mask assembly, and contour
  extraction with OpenCV
- Optional asynchronous mode, so the caller's thread is never blocked by a model run
- iOS 16.0+

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

Only models that are members of the target end up in the built framework. The current switch maps
`vistas_s`, `vistas_m`, and `vistas_l` to the v2 Vistas models; the YOLO11 COCO and BallAR cases
are commented out and their packages excluded from the target.

`YOLOUnity/Models/convert-coreml.py` exports an Ultralytics `.pt` model to Core ML
(`python convert-coreml.py model.pt --precision int8|fp16`).

## Building the framework

Build for a generic iOS device in Release:

```bash
xcodebuild -project YOLOUnity.xcodeproj -scheme YOLOUnity -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build build
```

The result is `build/Build/Products/Release-iphoneos/YOLOUnity.framework`. To use it in Unity, copy
it to `Assets/Plugins/iOS/` and enable it for iOS with **Add to Embedded Binaries** in the plugin
import settings.

## C API

All functions are exported with `@_cdecl` (see `YOLOUnity/YOLOBridge.swift`). Swift `Int` is 64-bit
and `Bool` is 1 byte, so declare them as `long` and `[MarshalAs(UnmanagedType.I1)] bool` in C#.

| Function | Purpose |
|---|---|
| `bool InitializeYOLO(const char* modelName, float confidenceThreshold, float iouThreshold, const char* scaleMethod)` | Loads a model. `scaleMethod` is `scaleFit`, `scaleFill`, or `centerCrop` (how Vision fits the image to the model input). Returns false if the model can't be loaded. |
| `void RegisterYOLOCallback(YOLOCallback callback)` | Sets the function that receives results. |
| `const uint8_t* GetClassNames(int32_t* outCount, int32_t* outByteLength)` | The model's class names, null-separated UTF-8, indexed by class index. Owned by the framework; valid until the next `InitializeYOLO`. |
| `void SetYOLOAsync(bool enabled)` | Off by default. See [Asynchronous mode](#asynchronous-mode). |
| `void RunYOLO_Byte(const uint8_t* bgra, long width, long height, uint64_t timestamp, float scaleX, float scaleY)` | Runs the model on an 8-bit BGRA image whose rows are bottom-up (as read back from a Unity texture). The pixels are copied before the call returns. |
| `void RunYOLO(const float* rgba, long width, long height, uint64_t timestamp, float scaleX, float scaleY)` | Same, for a float RGBA image with values in 0–1. |

`timestamp` is echoed back in the callback so you can match results to frames (`0` means "now", in
milliseconds since 1970). `scaleX` and `scaleY` multiply every returned coordinate, which lets you
run on a downscaled image and receive coordinates in the original image's pixels.

### Results callback

```c
typedef void (*YOLOCallback)(
    int32_t numDetections,
    const int32_t* classIndices,                // numDetections
    const float* scores,                         // numDetections
    const int32_t* boxes,                        // numDetections * 4: x1, y1, x2, y2
    const int32_t* points, int32_t pointsCount,  // all contour points, flattened as x, y
    const int32_t* indices, int32_t indicesCount,
    const int32_t* centroids,                    // numDetections * 2: x, y
    uint64_t timestamp);
```

`indices` says which points belong to which contour, as one run per detection: the run starts with
that detection's first point index, then lists the end index (exclusive) of each of its contours,
and finishes with `-1`. A detection with two contours looks like `[0, 40, 65, -1]`: contour 1 is
points 0–39 and contour 2 is points 40–64. Point `j` is `points[2j], points[2j + 1]`.

The pointers are only valid during the callback, so copy what you need. The callback is invoked on
a background queue, one call at a time and in submission order.

### Asynchronous mode

By default, `RunYOLO*` blocks the calling thread for the whole model run (tens of milliseconds for
the larger models). With `SetYOLOAsync(true)`:

- `RunYOLO*` copies the image and returns immediately; the model runs on a private serial queue.
- A frame submitted while the model is busy waits for it, replacing any older waiting frame. The
  model always picks up the newest frame next, and replaced frames never produce a callback.
- Post-processing runs on its own serial queue, so the next model run overlaps the previous frame's
  post-processing.

Submit every camera frame: throughput stays at one result per model run, and the frame waiting for
the model is never more than one camera frame old. Set the mode once, before the first `RunYOLO*`.

## License

The YOLOUnity code is licensed under the [MIT License](LICENSE).

The model files in `YOLOUnity/Models` are not covered by the MIT License. The Mapillary Vistas
models (`vistas-*`) are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
and may be used for non-commercial purposes only. The `yolo11*-seg` models are Ultralytics' pretrained
COCO models, distributed under Ultralytics' [AGPL-3.0 License](https://ultralytics.com/license).
