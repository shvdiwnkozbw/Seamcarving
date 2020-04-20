import sys
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Seam Carving')
    parser.add_argument('--RorC', default='c', type=str, 
                        help='Row or Colomn, input r or c')
    parser.add_argument('--scale', default=100, type=int,
                        help='Target scale')
    parser.add_argument('--image_in', default='./img/t001.png', type=str,
                        help='input image')
    parser.add_argument('--image_out', default='./image_out.png', type=str,
                        help='output image')
    args = parser.parse_args()
    return args

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # 我们计算红，绿，蓝通道中的能量值之和
    energy_map = convolved.sum(axis=2)

    return energy_map

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # 处理图像的左侧边缘，确保我们不会索引-1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j]= idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def main():
    args = parse_args()
    
    '''
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    
    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]
    '''
    which_axis = args.RorC
    scale = args.scale
    in_filename = args.image_in
    out_filename = args.image_out

    img = imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, scale)
        print('cropping row lines')
    elif which_axis == 'c':
        print('cropping colomn lines')
        out = crop_c(img, scale)
    
    '''Ploting'''
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(img)
#    plt.title('original')
#    plt.subplot(1,2,2)
#    plt.imshow(out)
#    plt.title('seam carved')
#    plt.show()
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()