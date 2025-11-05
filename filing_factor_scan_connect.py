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

def moving_flowparticle(img,m):
    img_ave=np.zeros((2048,2048))
    for i in range(0,m):
        img_ave=img_ave+img[i]/m
    if img_ave.dtype != np.uint8:
        img_ave = (255 * (img_ave / np.max(img_ave))).astype(np.uint8)
    img_ave = cv2.medianBlur(img_ave, ksize=3) 
    return img_ave

def hough_circle(img_ave):
    # Hough変換で円検出(param1:大きいほど検出しやすい,param2:小さいほど検出しやすい)
    circles = cv2.HoughCircles(
        img_ave, 
        cv2.HOUGH_GRADIENT, dp=1.0, minDist=10,
        param1=150, param2=15, minRadius=20, maxRadius=25
    )
    return circles

def two_value(img_ave,s):
    for i in range(2048):
        for j in range(2048):
            if img_ave[i,j]<s:
                img_ave[i,j]=0
            else:
                img_ave[i,j]=255
    return img_ave

def get_circlepx(circles):
    x = circles[0, :, 0]
    y = circles[0, :, 1]
    points = np.zeros((len(x), 2))
    points[:, 0] = x
    points[:, 1] = y
    points=points[np.argsort(points[:,1])]
    points_sort=np.zeros([len(points),2])
    j=0
    for i in range(int(len(points)/20)):
        points_tmp=points[j:j+20,:]
        points_tmp=points_tmp[np.argsort(points_tmp[:,0])]
        points_sort[j:j+20]=points_tmp
        j=j+20
    return points,x

def circle_line(img_ave,circles,cf):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            #cfフラグ0なら円周、中心1なら塗りつぶし
            if cf==0:
                cv2.circle(img_ave, center, radius, 0, 2)
                cv2.circle(img_ave, center, 2, 0, 3)
            else:
                cv2.circle(img_ave,center,radius,0,thickness=-1)
    return img_ave

def fill_judge_left(points,x,img_ave,img_ave2,black,white):
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

            #充填しているかの判定
            if (judge==1).all():    
                connected=is_connected_black_path_left(crop) #ピラー円間が輝度値0のpxでつながっているか判定、bool型で返す
                if connected==True:  
                    img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=0
                    black+=47**2
                else:
                    img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=150
                    white+=47**2
            else:
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0]):int(points[i,0])+47]=150
                white+=47**2
    return img_ave2,black,white

def fill_judge_right(points,x,img_ave,img_ave2,black,white):
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
            else:
                img_ave2[int(points[i,1]):int(points[i,1])+47,int(points[i,0])-47:int(points[i,0])]=150
                white+=47**2
    return img_ave2,black,white

def save_graph(t,e,filepath):
    #充填率の時間変化の散布図
    plt.scatter(t,e)
    plt.xlabel('t(min)')
    plt.ylabel('filing factor(-)')
    plt.title('t-filfac')
    plt.savefig(filepath+'/imgana/filing_factor/filing_factor.png')

    #csv化  
    forcsv=pd.DataFrame({"time[min]":t,"filing factor[-]":e})
    forcsv.to_csv(filepath+'/imgana/filing_factor/filing_factor.png')

def main():
    m=50    #連続画像枚数
    n=24    #解析スタック数
    s=150    #2値化閾値（ピラー円と粒子の輝度値のみが0になる値に設定）
    t=range(4,26) #画像撮影時間(min)
    e=np.zeros(n)  #充填率格納箱
    filepath="F:/experiment/2025_08_18_grid_particle"
    for k in range(4,26):
        img=tiff.imread(filepath+f"/{k}.tiff")

        #連続画像の平均を取り，流動粒子を排除
        img_ave=moving_flowparticle(img,m)

        cv2.imwrite(filepath+f"/imgana/filing_factor/ave{k}.tiff",img_ave)

        circles=hough_circle(img_ave)

        img_ave=two_value(img_ave,s)

        cv2.imwrite(filepath+f"/imgana/filing_factor/two_value{k}.tiff",img_ave)

        #2値化後の画像をグレースケールで読み込み
        img_ave=cv2.imread(filepath+f"/imgana/filing_factor/two_value{k}.tiff",cv2.IMREAD_GRAYSCALE)
        img_ave2=cv2.imread(filepath+f"/imgana/filing_factor/two_value{k}.tiff",cv2.IMREAD_GRAYSCALE)

        img_ave=circle_line(img_ave,circles,0)

        cv2.imwrite(filepath+f"/imgana/filing_factor/add_circle_line{k}.tiff", img_ave)

        img_ave=circle_line(img_ave,circles,1)

        #cv2.imwrite(f"F:/experiment/2025_06_21_grid_particle/imgana/filing_factor/add_circle{k}.tiff", img_ave)円周のみで十分か

        points,x=get_circlepx(circles)
    
        #充填率算出のための格納箱
        black=0
        white=0

        img_ave2,black,white=fill_judge_left(points,x,img_ave,img_ave2,black,white)         
        img_ave2,black,white=fill_judge_right(points,x,img_ave,img_ave2,black,white)

        #充填領域を黒く塗った画像の保存
        cv2.imwrite(filepath+f"/imgana/filing_factor/result{k}.tiff",img_ave2)

        #充填率の算出
        e[k-2]=black/(black+white)

    save_graph(t,e,filepath)

if __name__ == "__main__":
    main()