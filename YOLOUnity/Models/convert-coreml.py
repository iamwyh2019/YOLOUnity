from ultralytics import YOLO

for sz in ('n', 's', 'm', 'l', 'x'):
    model = YOLO(f'yolo11{sz}-seg.pt')
    model.export(format='coreml', int8=True, imgsz=640)