from ultralytics import YOLO

# for sz in ('n', 's', 'm', 'l', 'x'):
#     model = YOLO(f'yolo11{sz}-seg.pt')
#     model.export(format='coreml', int8=True, imgsz=640)

for sz in ('m', 'l'):
    model = YOLO(f'vistas-{sz}.pt')
    model.export(data='data-vistas.yaml', format='coreml', int8=True, imgsz=640)