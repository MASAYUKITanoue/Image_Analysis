import numpy as np
import cv2
import pandas as pd
import tifffile as tiff

m=50
target_num=target_num=[1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,29]

for j in target_num:
    # 元画像読み込み
    gray = tiff.imread(f"F:/experiment/2025_07_05_grid_particle/imgana/numden/time/crop{j}.tiff")
  
    #連続画像の平均を取り動きのある粒子を排除
    img_ave=np.zeros(gray[0].shape, dtype=np.float32)

    for i in range(0,m):
        img_ave=img_ave+gray[i]/m

    if img_ave.dtype != np.uint8:
        img_ave = (255 * (img_ave / np.max(img_ave))).astype(np.uint8)

    tiff.imwrite(f"F:/experiment/2025_07_05_grid_particle/imgana/numden/time/ave/crop{j}_ave.tiff",img_ave)