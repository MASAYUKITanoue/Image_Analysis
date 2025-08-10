import numpy as np
import cv2
from scipy.spatial import Delaunay
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
from collections import deque

#ピラー間で粒子が繋がっているかを判定する関数（左上）
def is_connected_black_path_left(img):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)

    # 入口または出口が黒でない（通れない）なら即 False
    if img[0, 0] != 0 or img[h-1, w-1] != 0:
        return False

    # BFS開始
    queue = deque()
    queue.append((0, 0))
    visited[0, 0] = True

    # 4近傍（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()

        # ゴール到達判定
        if (y, x) == (h - 1, w - 1):
            return True

        # 隣接ピクセルをチェック
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and img[ny, nx] == 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return False

#ピラー間で粒子が繋がっているかを判定する関数（右上）
def is_connected_black_path_right(img):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)

    # 入口または出口が黒でない（通れない）なら即 False
    if img[0, w-1] != 0 or img[h-1, 0] != 0:
        return False

    # BFS開始
    queue = deque()
    queue.append((0, w-1))
    visited[0, w-1] = True

    # 4近傍（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()

        # ゴール到達判定
        if (y, x) == (h - 1, 0):
            return True

        # 隣接ピクセルをチェック
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and img[ny, nx] == 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return False

m=50    #連続画像枚数
n=24    #解析スタック数
s=150    #2値化閾値（ピラー円と粒子の輝度値のみが0になる値に設定）
t=range(2,26) #画像撮影時間
e=np.zeros(n)  #充填率格納箱

img=tiff.imread(f"F:/experiment/2025_08_02_grid_particle/after.tiff")

#連続画像の平均を取り，流動粒子を排除
img_ave=np.zeros((2048,2048))

for i in range(0,m):
    img_ave=img_ave+img[i]/m

if img_ave.dtype != np.uint8:
    img_ave = (255 * (img_ave / np.max(img_ave))).astype(np.uint8)

#流動粒子排除画像の保存
cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/ave.tiff",img_ave)

# メディアンフィルタ適用（）
img_ave = cv2.medianBlur(img_ave, ksize=3) 

# Hough変換で円検出(param1:大きいほど検出しやすい,param2:小さいほど検出しやすい)
circles = cv2.HoughCircles(
    img_ave, 
    cv2.HOUGH_GRADIENT, dp=1.0, minDist=10,
    param1=150, param2=20, minRadius=20, maxRadius=25
)

#閾値sを基準に2値化
for i in range(2048):
    for j in range(2048):
        if img_ave[i,j]<s:
            img_ave[i,j]=0
        else:
            img_ave[i,j]=255

#2値化後の画像の保存
cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/two_value.tiff",img_ave)

#2値化後の画像をグレースケールで読み込み
img_ave=cv2.imread(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/two_value.tiff",cv2.IMREAD_GRAYSCALE) #判定用
img_ave2=cv2.imread(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/two_value.tiff",cv2.IMREAD_GRAYSCALE)    #出力用

# 円の中心座標を取得
x = circles[0, :, 0]
y = circles[0, :, 1]

# 円を描画
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(img_ave, center, radius, 0, 2)
        cv2.circle(img_ave, center, 2, 0, 3)

cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/add_circle.tiff", img_ave)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(img_ave,center,radius,0,thickness=-1)

cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/add_black_circle.tiff", img_ave)

# 円中心座標を配列に格納
points = np.zeros((len(x), 2))
points[:, 0] = x
points[:, 1] = y
    
#充填率算出のための格納箱
black=0
white=0

#ピラー円中心を左上とした矩形内の充填判定
for i in range(len(x)):
    if points[i,0]+47<=points[:,0].max()+8 and points[i,1]+47<=points[:,1].max()+8: #ピラー領域内の矩形のみを判定対象とする条件
        crop=img_ave[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47] #矩形領域の取り出し

        #矩形を反時計回りに45度回転
        center=(47//2,47//2)
        M=cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated=cv2.warpAffine(crop, M, (47, 47))

        #矩形の各列においていずれかの行に輝度値0のpxがあるかを判定
        judge=np.zeros(47) 
        for j in range(47):
            if (rotated[:,j]<=0).any():
                judge[j]=1  #trueの場合，judgeに1を格納する

        #judgeのすべての値が1なら，その矩形を黒く塗る
        if (judge==1).all():  
            connected=is_connected_black_path_left(crop) #ピラー円間が輝度値0のpxでつながっているか判定、bool型で返す
            if connected==True:  
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=0
                black+=47**2
            else:
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=150
                white+=47**2
                cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/test/left/crop{i}.tiff",crop)
        else:
            img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=150
            white+=47**2
            cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/test/left/crop{i}.tiff",crop)
            

#ピラー円中心を右上とした矩形内の充填判定
for i in range(len(x)):
    if points[i,0]-47>=points[:,0].min()-8 and points[i,1]+47<=points[:,1].max()+8: #ピラー領域内の矩形のみを判定対象とする条件
        crop=img_ave[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])] #矩形領域の取り出し
        
        #矩形を反時計回りに45度回転
        center=(47//2,47//2)
        M=cv2.getRotationMatrix2D(center, 135, 1.0)
        rotated=cv2.warpAffine(crop, M, (47, 47))

        #矩形の各列においていずれかの行に輝度値0のpxがあるかを判定
        true=np.zeros(47)
        for j in range(47):
            if (rotated[:,j]<=50).any():
                true[j]=1   #trueの場合，judgeに1を格納する

        #judgeのすべての値が1なら，その矩形を黒く塗る      
        if (true==1).all():
            connected=is_connected_black_path_right(crop) #ピラー円間が輝度値0のpxでつながっているか判定、bool型で返す
            if connected==True:
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]=0
                black+=47**2
            else:
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]=150
                white+=47**2
                cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/test/right/crop{i}.tiff",crop)
        else:
            img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]=150
            white+=47**2
            cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/test/right/crop{i}.tiff",crop)

#充填領域を黒く塗った画像の保存
cv2.imwrite(f"F:/experiment/2025_08_02_grid_particle/imgana/filing_factor/result.tiff",img_ave2)

print(black)
print(white)
#充填率の算出
    #e[k-2]=black/(black+white)

#充填率の時間変化の散布図
#plt.scatter(t,e)
#plt.xlabel('t(min)')
#plt.ylabel('filing factor(-)')
#plt.title('t-filfac')
#plt.savefig('F:/experiment/2025_06_21_grid_particle/imgana/filing_factor/filing_factor.png')

#csv化
#forcsv=pd.DataFrame({"time[min]":t,"filing factor[-]":e})
#forcsv.to_csv('F:/experiment/2025_06_21_grid_particle/imgana/filing_factor/filing_factor.png')

