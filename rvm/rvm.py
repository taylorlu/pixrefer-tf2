import torch
import argparse, os

model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", skip_validation=True)
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter", skip_validation=True)

parser = argparse.ArgumentParser(description='Human segmentation and matting')
parser.add_argument("-i", '--inputpath', type=str, help='input path of video')
args = parser.parse_args()

print(args.inputpath)
output = os.path.basename(args.inputpath).split('.')[0]
convert_video(model, 
    input_source=args.inputpath,     # A video file or an image sequence directory.
    input_resize=(1920, 1080),       # [Optional] Resize the input (also the output).
    downsample_ratio=0.25,           # [Optional] If None, make downsampled max size be 512px.
    output_type='png_sequence',      # Choose "video" or "png_sequence"
    output_composition=output,       # File path if video; directory path if png sequence.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    num_workers=0,                   # Only for image sequence input. Reader threads.
    device='cuda',
    progress=True)
