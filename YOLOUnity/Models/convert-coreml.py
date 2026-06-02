import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Export a YOLO .pt model to Core ML.')
parser.add_argument('pt_path', help='Path to the .pt model file')
parser.add_argument('--precision', choices=('fp16', 'int8'), default='int8',
                    help='fp16: FP16 weights. int8: 8-bit k-means weight palettization (data-free).')
parser.add_argument('--imgsz', type=int, default=640)
args = parser.parse_args()

export_kwargs = {'format': 'coreml', 'imgsz': args.imgsz}
if args.precision == 'fp16':
    export_kwargs['half'] = True
else:
    export_kwargs['int8'] = True

YOLO(args.pt_path).export(**export_kwargs)
