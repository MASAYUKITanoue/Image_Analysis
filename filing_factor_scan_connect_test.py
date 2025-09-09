import numpy as np
import cv2
from scipy.spatial import Delaunay
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
from collections import deque

m=150    #連続画像枚数
n=8    #解析スタック数
s=150    #2値化閾値（ピラー円と粒子の輝度値のみが0になる値に設定）
t=np.array((4,8,12,16,20,24,28,31)) #画像撮影時間
e=np.zeros(n)  #充填率格納箱

#ROI内の流動有無判定
def is_flowing_in_roi(stack_3d, y0, y1, x0, x1, k=4.0, smooth=3, min_ratio=0.002):
    sub = stack_3d[:, y0:y1, x0:x1].astype(np.float32)
    T = sub.shape[0]
    if T < 2:
        return False
    moving_ratios = []
    for ti in range(1, T):
        prev = sub[ti-1]; curr = sub[ti]
        diff = np.abs(curr - prev)
        if smooth and smooth > 0:
            diff = cv2.GaussianBlur(diff, (smooth, smooth), 0)
        med = np.median(diff)
        mad = np.median(np.abs(diff - med)) + 1e-8
        thr = med + k*1.4826*mad
        mask = (diff > thr)
        moving_ratios.append(mask.mean())
    return np.mean(moving_ratios) > min_ratio

#ピラー間で粒子が繋がっているかを判定する関数（左上）
def is_connected_black_path_left(img):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)
    if img[0, 0] != 0 or img[h-1, w-1] != 0:
        return False
    queue = deque(); queue.append((0, 0)); visited[0, 0] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        if (y, x) == (h-1, w-1):
            return True
        for dy, dx in directions:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and img[ny, nx] == 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return False

#ピラー間で粒子が繋がっているかを判定する関数（右上）
def is_connected_black_path_right(img):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)
    if img[0, w-1] != 0 or img[h-1, 0] != 0:
        return False
    queue = deque(); queue.append((0, w-1)); visited[0, w-1] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        if (y, x) == (h-1, 0):
            return True
        for dy, dx in directions:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and img[ny, nx] == 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return False

for idx,k in enumerate(t):
    # 連続フレーム読み込み
    img=tiff.imread(f"F:/experiment/2025_08_18_grid_particle/{k}.tiff")

    #連続画像の平均を取り，流動粒子を排除
    img_ave=np.zeros((2048,2048))
    for i in range(0,m):
        img_ave=img_ave+img[i]/m
    if img_ave.dtype != np.uint8:
        img_ave = (255 * (img_ave / np.max(img_ave))).astype(np.uint8)
    cv2.imwrite(f"F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/ave/ave{k}.tiff",img_ave)

    #フィルタと円検出
    img_ave = cv2.medianBlur(img_ave, ksize=3) 
    circles = cv2.HoughCircles(img_ave, cv2.HOUGH_GRADIENT, dp=1.0, minDist=10,
        param1=150, param2=15, minRadius=20, maxRadius=25)

    #閾値sで二値化
    for i in range(2048):
        for j in range(2048):
            if img_ave[i,j]<s:
                img_ave[i,j]=0
            else:
                img_ave[i,j]=255
    cv2.imwrite(f"F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/binary/binary{k}.tiff",img_ave)

    img_ave=cv2.imread(f"F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/binary/binary{k}.tiff",cv2.IMREAD_GRAYSCALE)
    img_ave2=img_ave.copy()

    # 円の中心座標
    x = circles[0, :, 0]; y = circles[0, :, 1]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img_ave,(i[0],i[1]),i[2],0,2)
            cv2.circle(img_ave,(i[0],i[1]),2,0,3)
    cv2.imwrite(f"F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/add_circle/add_circle{k}.tiff", img_ave)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img_ave,(i[0],i[1]),i[2],0,thickness=-1)
    points=np.zeros((len(x),2)); points[:,0]=x; points[:,1]=y
    
    black=0; white=0

    #左上矩形判定
    for i in range(len(x)):
        if points[i,0]+47<=points[:,0].max()+8 and points[i,1]+47<=points[:,1].max()+8:
            crop=img_ave[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]
            center=(47//2,47//2); M=cv2.getRotationMatrix2D(center,45,1.0)
            rotated=cv2.warpAffine(crop,M,(47,47))
            judge=np.zeros(47) 
            for j in range(47):
                if (rotated[:,j]<=0).any():
                    judge[j]=1  
            if (judge==1).all():    
                connected=is_connected_black_path_left(crop)
                if connected==True:  
                    img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=0
                    black+=47**2
                else:
                    y0,y1=int(points[i,1]),int(points[i,1])+47
                    x0,x1=int(points[i,0]),int(points[i,0])+47
                    flowing=is_flowing_in_roi(img,y0,y1,x0,x1)
                    if flowing:
                        img_ave2[y0:y1,x0:x1]=150; white+=47**2
                    else:
                        img_ave2[y0:y1,x0:x1]=0; black+=47**2
            else:
                y0,y1=int(points[i,1]),int(points[i,1])+47
                x0,x1=int(points[i,0]),int(points[i,0])+47
                flowing=is_flowing_in_roi(img,y0,y1,x0,x1)
                if flowing:
                    img_ave2[y0:y1,x0:x1]=150; white+=47**2
                else:
                    img_ave2[y0:y1,x0:x1]=0; black+=47**2

    #右上矩形判定
    for i in range(len(x)):
        if points[i,0]-47>=points[:,0].min()-8 and points[i,1]+47<=points[:,1].max()+8:
            crop=img_ave[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]
            center=(47//2,47//2); M=cv2.getRotationMatrix2D(center,135,1.0)
            rotated=cv2.warpAffine(crop,M,(47,47))
            true=np.zeros(47)
            for j in range(47):
                if (rotated[:,j]<=50).any():
                    true[j]=1
            if (true==1).all():
                connected=is_connected_black_path_right(crop)
                if connected==True:  
                    img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]=0
                    black+=47**2
                else:
                    y0,y1=int(points[i,1]),int(points[i,1])+47
                    x0,x1=int(points[i,0])-47,int(points[i,0])
                    flowing=is_flowing_in_roi(img,y0,y1,x0,x1)
                    if flowing:
                        img_ave2[y0:y1,x0:x1]=150; white+=47**2
                    else:
                        img_ave2[y0:y1,x0:x1]=0; black+=47**2
            else:
                y0,y1=int(points[i,1]),int(points[i,1])+47
                x0,x1=int(points[i,0])-47,int(points[i,0])
                flowing=is_flowing_in_roi(img,y0,y1,x0,x1)
                if flowing:
                    img_ave2[y0:y1,x0:x1]=150; white+=47**2
                else:
                    img_ave2[y0:y1,x0:x1]=0; black+=47**2

    cv2.imwrite(f"F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/output/result{k}.tiff",img_ave2)
    e[idx]=black/(black+white)

plt.scatter(t,e)
plt.xlabel('t(min)'); plt.ylabel('filing factor(-)'); plt.title('t-filfac')
plt.savefig('F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/output/filing_factor.png')
forcsv=pd.DataFrame({"time[min]":t,"time":t*60,"filing factor[-]":e,"porosity":1-e})
forcsv.to_csv('F:/experiment/2025_08_18_grid_particle/imgana/filing_factor/output/filing_factor.csv')
