# ``YOLOUnity``

Runs YOLO instance-segmentation models with Core ML and returns detections through a C API.

## Overview

YOLOUnity loads a Core ML YOLO segmentation model, runs it through Vision on the CPU and Neural
Engine, and post-processes the output on device: confidence filtering, per-class non-maximum
suppression, mask assembly, and contour extraction with OpenCV. Results (class, score, box,
contour polygons, centroid) are delivered to a registered C callback, which makes the framework
easy to call from Unity or another native host.

The C API is defined in `YOLOBridge.swift` and documented in the repository's README.
