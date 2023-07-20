from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
from os import listdir
# get the path/directory
folder_dir0 = "datasets/test/rain100norain/"
p0 = 0
s0 = 0
folder_dir1 = "datasets/test/rain100norain/"
p1 = 0
s1 = 0
folder_dir2 = "datasets/test/test12norain/"
p2 = 0
s2 = 0
for images in os.listdir(folder_dir0):
    # check if the image ends with png
    if (images.endswith(".png")):
        print(images[7:10])
        img1 = cv2.imread('datasets/test/rain100norain/{}'.format(images))
        img2 = cv2.imread('results/Rain100H/Model129/{}'.format(images[2:]))
        # print(img2.shape)
        # print(img1.shape)
        p0 = p0 + compare_psnr(img1,img2)
        s0 = s0 + compare_ssim(img1, img2,channel_axis = 2)

for images in os.listdir(folder_dir1):
    # check if the image ends with png
    if (images.endswith(".png")):
        print(images[7:10])
        img1 = cv2.imread('datasets/test/rain100norain/{}'.format(images))
        img2 = cv2.imread('results/Rain100L/Model129/{}'.format(images[2:]))
        # print(img2.shape)
        p1 = p1 + compare_psnr(img1, img2)
        s1 = s1 + compare_ssim(img1, img2, channel_axis=2)
for images in os.listdir(folder_dir2):
    # check if the image ends with png
    if (images.endswith(".png")):
        print(images)
        img1 = cv2.imread('datasets/test/test12norain/{}'.format(images))
        img2 = cv2.imread('results/test12/Model129/{}'.format(images))
        #print(img2.shape)
        p2 = p2 + compare_psnr(img1, img2)
        s2 = s2 + compare_ssim(img1, img2, channel_axis=2)

print('->Rain100H-PSNR:{:.4f}-SSIM:{:.4f}\n->Rain100L-PSNR:{:.4f}-SSIM:{:.4f}\n-> Test12 -PSNR:{:.4f}-SSIM:{:.4f}'.format(p0/100,s0/100,p1/100,s1/100,p2/12,s2/12))
