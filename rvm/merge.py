import argparse, os, cv2
import numpy as np

parser = argparse.ArgumentParser(description='merge head/pixrefer and original whole body image')
parser.add_argument("-p", '--headpath', type=str, help='path of head/pixrefer folder')
parser.add_argument("-b", '--bodypath', type=str, help='path of original whole body folder')
parser.add_argument("-s", '--size', type=int, default=512, help='crop size of head')
args = parser.parse_args()

bodypath = args.bodypath
headpath = args.headpath
outputpath = bodypath+'_final'
os.makedirs(outputpath, exist_ok=True)

center_x = 1000
center_y = 320

names = os.listdir(headpath)
for name in names:
    head_rgba = cv2.imread(os.path.join(headpath, name), cv2.IMREAD_UNCHANGED)
    body_rgba = cv2.imread(os.path.join(bodypath, name), cv2.IMREAD_UNCHANGED)
    body_rgba[..., :3] = (body_rgba[..., :3] * (body_rgba[..., 3:]/255).astype(np.float)).astype(np.uint8)

    body_rgba[center_y-args.size//2: center_y-args.size//2+args.size, 
              center_x-args.size//2: center_x-args.size//2+args.size,
              :] = head_rgba

    cv2.imwrite(os.path.join(outputpath, name), body_rgba)
