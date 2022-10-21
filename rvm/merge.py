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

names = os.listdir(headpath)

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

headcoord = np.load('headcoordsize.npy')

for name in names:
    head_rgba = cv2.imread(os.path.join(headpath, name), cv2.IMREAD_UNCHANGED)
    body_rgba = cv2.imread(os.path.join(bodypath, '{:04d}.png'.format(idx[int(name.split('.')[0])])), cv2.IMREAD_UNCHANGED)
    body_rgba[..., :3] = (body_rgba[..., :3] * (body_rgba[..., 3:]/255).astype(np.float)).astype(np.uint8)

    body_rgba[headcoord[2]:headcoord[3], headcoord[0]:headcoord[1]] = cv2.resize(head_rgba, (headcoord[1]-headcoord[0], headcoord[3]-headcoord[2]))

    cv2.imwrite(os.path.join(outputpath, name), body_rgba)
