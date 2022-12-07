import torch, cv2
import argparse, os

local_hub = os.path.join(torch.hub.get_dir(), 'PeterL1n_RobustVideoMatting_master')
if(os.path.exists(local_hub)):
    model = torch.hub.load(local_hub, "resnet50", source='local')
    convert_video = torch.hub.load(local_hub, "converter", source='local')
else:
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", skip_validation=True)
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter", skip_validation=True)

parser = argparse.ArgumentParser(description='Human segmentation and matting')
parser.add_argument("-i", '--inputpath', type=str, help='input path of video')
parser.add_argument("-d", '--downsample_ratio', default=0.25, type=float, help='downsample ratio, (0.25/portrait, 0.4/full body)')
args = parser.parse_args()

print(args.inputpath)
if(os.path.isdir(args.inputpath)):
    img = cv2.imread(os.path.join(args.inputpath, os.listdir(args.inputpath)[0]))
    input_resize = (img.shape[1], img.shape[0])
else:
    cap = cv2.VideoCapture(args.inputpath)
    _, frame = cap.read()
    input_resize = (frame.shape[1], frame.shape[0])

output = os.path.basename(args.inputpath).split('.')[0]+'_final2'
convert_video(model, 
    input_source=args.inputpath,            # A video file or an image sequence directory.
    input_resize=input_resize,              # [Optional] Resize the input (also the output).
    downsample_ratio=args.downsample_ratio, # [Optional] If None, make downsampled max size be 512px.
    output_type='png_sequence',             # Choose "video" or "png_sequence"
    output_composition=output,              # File path if video; directory path if png sequence.
    seq_chunk=12,                           # Process n frames at once for better parallelism.
    num_workers=0,                          # Only for image sequence input. Reader threads.
    device='cuda',
    progress=True)
