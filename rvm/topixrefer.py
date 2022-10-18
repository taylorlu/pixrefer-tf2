import argparse, os, cv2
import numpy as np

parser = argparse.ArgumentParser(description='make dataset for pixrefer')
parser.add_argument("-i", '--inputpath', type=str, help='path of crop folder')
parser.add_argument("-d", '--decainputpath', type=str, help='path of deca result folder')
args = parser.parse_args()

deca_output = args.decainputpath
output = args.inputpath+'_pixrefer'
os.makedirs(output, exist_ok=True)

names = os.listdir(args.inputpath)
for name in names:
    base = name.split('.')[0]
    rgba = cv2.imread(os.path.join(args.inputpath, name), cv2.IMREAD_UNCHANGED)
    rgb = rgba[..., :3]
    alpha = np.tile(rgba[..., 3:], [1, 1, 3])
    render = cv2.imread(os.path.join(deca_output, f'{base}/orig_{base}_shape_images.jpg'))
    render[-20:, :, :] = (rgb * (alpha/255))[-20:, :, :].astype(np.uint8)
    render[:, :20, :] = (rgb * (alpha/255))[:, :20, :].astype(np.uint8)
    render[:, -20:, :] = (rgb * (alpha/255))[:, -20:, :].astype(np.uint8)
    final = np.concatenate([rgb, render, alpha], axis=1)
    cv2.imwrite(os.path.join(output, name), final)
