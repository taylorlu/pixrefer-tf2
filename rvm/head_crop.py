import argparse, os, cv2
import numpy as np
from mtcnn import MTCNN

parser = argparse.ArgumentParser(description='head crop of png')
parser.add_argument("-i", '--rvmpath', type=str, help='path of rvm output folder (png with alpha)')
parser.add_argument("-s", '--scale', type=float, default=1.5, help='head crop rescale of mtcnn')
args = parser.parse_args()

fixed_head_size = 512
rootpath = args.rvmpath
output_crop = rootpath+'_crop'
os.makedirs(output_crop, exist_ok=True)

def rect2square(face):
    center = [int(face[0] + face[2]/2), int(face[1] + face[3]/2)]
    lt_x = int(center[0] - face[2]/2*args.scale)
    rd_x = int(center[0] + face[2]/2*args.scale)
    lt_y = int(center[1] - face[3]/2*args.scale)
    rd_y = int(center[1] + face[3]/2*args.scale)

    if(rd_y-lt_y>rd_x-lt_x):
        border = rd_y-lt_y
    else:
        border = rd_x-lt_x

    lt_x = int(center[0] - border//2)
    rd_x = int(center[0] + border//2)
    lt_y = int(center[1] - border//2)
    rd_y = int(center[1] + border//2)
    return lt_x, rd_x, lt_y, rd_y

max_x, max_y = 0, 0
min_x, min_y = 10000, 10000
bound_box = []
max_face_x, max_face_y = 0, 0
min_face_x, min_face_y = 10000, 10000
detector = MTCNN()
for name in os.listdir(rootpath):
    img = cv2.imread(os.path.join(rootpath, name), cv2.IMREAD_UNCHANGED)

    face = detector.detect_faces(img[..., :3])[0]['box']
    lt_x, rd_x, lt_y, rd_y = rect2square(face)
    if(lt_x<min_face_x):
        min_face_x = lt_x
    if(rd_x>max_face_x):
        max_face_x = rd_x
    if(lt_y<min_face_y):
        min_face_y = lt_y
    if(rd_y>max_face_y):
        max_face_y = rd_y

    gray = img[..., 3:]
    thresh = np.where(gray>10, 255, 0).astype(np.uint8)
    kernel = np.ones((3,3), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    if(x<min_x or lt_x<min_x):
        min_x = min(x, lt_x)
    if(y<min_y or lt_y<min_y):
        min_y = min(y, lt_y)
    if(x+w>max_x or rd_x>max_x):
        max_x = max(x+w, rd_x)
    if(y+h>max_y or rd_y>max_y):
        max_y = max(y+h, rd_y)


for i, name in enumerate(os.listdir(rootpath)):
    img = cv2.imread(os.path.join(rootpath, name), cv2.IMREAD_UNCHANGED)
    head = img[min_face_y:max_face_y, min_face_x:max_face_x, :]
    head = cv2.resize(head, (fixed_head_size, fixed_head_size))
    img = img[min_y:max_y, min_x:max_x]
    cv2.imwrite(os.path.join(rootpath, name), img)
    cv2.imwrite(os.path.join(output_crop, name), head)

os.system('ffmpeg -r 25 -i {}/%04d.png -y {}.mp4'.format(output_crop, output_crop))
os.system('ffmpeg -r 25 -i {}/%04d.png -y {}.mp4'.format(rootpath, rootpath))
np.save('headcoordsize.npy', np.array([min_face_x-min_x, max_face_x-min_x, min_face_y-min_y, max_face_y-min_y]))

# headcoord = np.load('headcoordsize.npy')
# img = cv2.imread('111/0000.png')
# head = cv2.imread('111_crop/0000.png')
# img[headcoord[2]:headcoord[3], headcoord[0]:headcoord[1]] = cv2.resize(head, (headcoord[1]-headcoord[0], headcoord[3]-headcoord[2]))
# cv2.imwrite('0.png', img)