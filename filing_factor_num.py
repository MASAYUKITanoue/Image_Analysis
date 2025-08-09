import numpy as np
import cv2
from scipy.spatial import Delaunay
import pandas as pd
import tifffile as tiff

#連続画像枚数
m=50
e=np.zeros(24)
t=np.array(range(1,25))

for k in range(2,26):

    # 元画像読み込み
    img_path = f"F:/experiment/2025_06_21_grid_particle/{k}.tiff"
    gray = tiff.imread(img_path)
    gray_mesh=tiff.imread(img_path)

    #連続画像の平均を取り動きのある粒子を排除
    img_ave=np.zeros((2048,2048))

    for i in range(0,m):
        img_ave=img_ave+gray[i]/m

    if img_ave.dtype != np.uint8:
        img_ave = (255 * (img_ave / np.max(img_ave))).astype(np.uint8)
    
    #連続画像の平均を取り動きのある粒子を排除
    img_ave_mesh=np.zeros((2048,2048))

    for i in range(0,m):
        img_ave_mesh=img_ave_mesh+gray_mesh[i]/m

    if img_ave_mesh.dtype != np.uint8:
        img_ave_mesh = (255 * (img_ave_mesh / np.max(img_ave_mesh))).astype(np.uint8)

    gray=img_ave
    gray_mesh=img_ave_mesh

    # Hough変換で円検出
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
        param1=150, param2=20, minRadius=20, maxRadius=25
    )

    # 円の中心座標を取得
    x = circles[0, :, 0]
    y = circles[0, :, 1]

    # 円を描画
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(gray, center, radius, 255, 2)
            cv2.circle(gray, center, 2, 255, 3)

    cv2.imwrite(f"F:/experiment/2025_06_21_grid_particle/imgana/filfac/detect{k}.tiff", gray)

    # 円中心座標を配列に格納
    points = np.zeros((len(x), 2))
    points[:, 0] = x
    points[:, 1] = y

    #パラメータ設定
    xmin=points[:,0].min()
    xmax=points[:,0].max()
    ymin=points[:,1].min()
    ymax=points[:,1].max()
    w=47    #正方形幅
    h=47    #正方形高さ
    r=50    #輝度値閾値
    n=80    #閾値を満たしたpx個数

    #元画像の読み込み
    img = gray
    img_mesh = gray_mesh

    
    #ピラー円中心を左上とした正方形の検査
    for i in points:
        x0=i[0]
        y0=i[1]
        x1=i[0]
        y1=i[1]+h
        x2=i[0]+w
        y2=i[1]+h
        x3=i[0]+w
        y3=i[1]
        #ピラー部以外の部分へのはみ出しを防ぐ条件
        if x2<=xmax+2 and y1<=ymax+2:
            pts=np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]],dtype=np.int32)
            cv2.polylines(img_mesh, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
            mask=np.zeros_like(img)
            cv2.fillPoly(mask,[pts],255)
            masked_pixels=img[mask==255]
            #輝度値r以下のpx数がn以上の場合黒く塗る
            count=np.sum(masked_pixels<=r)
            if count>=n:
                img[mask==255]=0
            else:
                img[mask==255]=255
   
        else:
            continue

    #ピラー円中心を右上とした正方形の検査
    for i in points:
        x0=i[0]
        y0=i[1]
        x1=i[0]-w
        y1=i[1]
        x2=i[0]-w
        y2=i[1]+h
        x3=i[0]
        y3=i[1]+h
        #ピラー部以外の部分へのはみ出しを防ぐ条件
        if x2>=xmin-2 and y2<=ymax+2:
            pts=np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]],dtype=np.int32)
            cv2.polylines(img_mesh, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
            mask=np.zeros_like(img)
            cv2.fillPoly(mask,[pts],255)
            masked_pixels=img[mask==255]
            #輝度値r以下のpx数がn以上の場合黒く塗る
            count=np.sum(masked_pixels<=r)
            if count>=n:
                img[mask==255]=0
            else:
                img[mask==255]=255
     
        else:
            continue
    e[k-2]=np.sum(img==0)/(np.sum(img==255)+np.sum(img==0))
    cv2.imwrite(f"F:/experiment/2025_06_21_grid_particle/imgana/filfac/mesh{k}.tiff", img_mesh)
    cv2.imwrite(f"F:/experiment/2025_06_21_grid_particle/imgana/filfac/result{k}.tiff", img)

result = pd.DataFrame({"time[min]": t,"filing_factor[-]": e})

result.to_csv("F:/experiment/2025_06_21_grid_particle/imgana/filfac/result.csv")
