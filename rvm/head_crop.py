import argparse, os, cv2
import numpy as np

parser = argparse.ArgumentParser(description='head crop of png')
parser.add_argument("-i", '--videopath', type=str, help='path of video.mp4')
parser.add_argument("-p", '--position', type=str, help='position of center, x,y')
parser.add_argument("-s", '--size', type=int, default=512, help='crop size of head')
args = parser.parse_args()

rootpath = os.path.basename(args.videopath).split('.')[0]
output_crop = rootpath+'_crop'
os.makedirs(output_crop, exist_ok=True)

center_x = int(args.position.split(',')[0])
center_y = int(args.position.split(',')[1])

cap = cv2.VideoCapture(args.videopath)
index = 0
while(cap.isOpened()):
    _, frame = cap.read()
    if(frame is None):
        break
    name = '{:04d}.png'.format(index)
    img = cv2.imread(os.path.join(rootpath, name), cv2.IMREAD_UNCHANGED)

    orig_img = np.concatenate([frame, img[..., 3:]], axis=-1)
    img = orig_img[center_y-args.size//2: center_y-args.size//2+args.size, 
                   center_x-args.size//2: center_x-args.size//2+args.size,
                   :]
    cv2.imwrite(os.path.join(rootpath, name), orig_img)
    cv2.imwrite(os.path.join(output_crop, name), img)
    index += 1

os.system('ffmpeg -r 25 -i {}/%04d.png -y {}.mp4'.format(output_crop, output_crop))
