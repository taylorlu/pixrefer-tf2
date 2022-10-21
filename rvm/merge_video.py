import argparse, os, cv2
import numpy as np

parser = argparse.ArgumentParser(description='merge head/pixrefer and original whole body image')
parser.add_argument("-p", '--headpath', type=str, help='path of head/pixrefer folder')
parser.add_argument("-b", '--bodypath', type=str, help='path of original whole body folder')
parser.add_argument("-v", '--videopath', type=str, help='path of background video')
parser.add_argument("-c", '--center', type=str, help='y,x center of fore in background video')
parser.add_argument("-s", '--size', type=int, default=512, help='crop size of head')
parser.add_argument("-l", '--scale', type=float, default=0.5, help='scale of fore')
args = parser.parse_args()

bodypath = args.bodypath
headpath = args.headpath
videopath = args.videopath
outputpath = bodypath+'_final'
os.makedirs(outputpath, exist_ok=True)

headcoord = np.load('headcoordsize.npy')
center_x2 = int(args.center.split(',')[1])
center_y2 = int(args.center.split(',')[0])

cap = cv2.VideoCapture(videopath)
names = os.listdir(headpath)
names2 = os.listdir(bodypath)

start = 6*25 #0
end = 12*25 #len(names2)
trunk = len(names)//(end-start-1) + 1
fwd = np.arange(start, end-1)
bck = np.arange(end-1, start, -1)
idx = np.array([]).astype(int)
for t in range(trunk):
    if(t%2==0):
        idx = np.concatenate([idx, fwd])
    else:
        idx = np.concatenate([idx, bck])

for i, name in enumerate(names):

    head_rgba = cv2.imread(os.path.join(headpath, name), cv2.IMREAD_UNCHANGED)
    body_rgba = cv2.imread(os.path.join(bodypath, '{:04d}.png'.format(idx[i])), cv2.IMREAD_UNCHANGED)
    body_rgba[..., :3] = (body_rgba[..., :3] * (body_rgba[..., 3:]/255).astype(np.float)).astype(np.uint8)

    _, frame = cap.read()
    _, frame = cap.read()
    if(frame is None):
        break

    body_rgba[headcoord[2]:headcoord[3], headcoord[0]:headcoord[1]] = cv2.resize(head_rgba, (headcoord[1]-headcoord[0], headcoord[3]-headcoord[2]))
    body_rgba = cv2.resize(body_rgba, (int(body_rgba.shape[1]*args.scale), int(body_rgba.shape[0]*args.scale)))

    y1 = center_y2 - body_rgba.shape[0]//2
    y2 = y1 + body_rgba.shape[0]
    x1 = center_x2 - body_rgba.shape[1]//2
    x2 = x1 + body_rgba.shape[1]

    body_rgba = body_rgba[:frame[y1:y2, x1:x2, :].shape[0], :frame[y1:y2, x1:x2, :].shape[1], :]
    frame[y1:y2, x1:x2, :] = (frame[y1:y2, x1:x2, :] * (1- body_rgba[..., 3:]/255.).astype(np.float) + body_rgba[..., :3] * (body_rgba[..., 3:]/255.).astype(np.float)).astype(np.uint8)

    cv2.imwrite(os.path.join(outputpath, name), frame)
