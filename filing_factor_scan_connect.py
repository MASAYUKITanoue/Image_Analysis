import numpy as np
import cv2
from scipy.spatial import Delaunay
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
from collections import deque
import os

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

def moving_flowparticle(img,height,width,m):
    img_ave=np.zeros((height,width))
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
        param1=120, param2=20, minRadius=20, maxRadius=25
    )
    return circles

def two_value(img_ave,height,width,s):
    for i in range(height):
        for j in range(width):
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
    #左上ピラーを0番として各行に番号付け
    points=points[np.argsort(points[:,1])]
    points_sort=np.zeros([len(points),2])
    j=0
    for i in range(int(len(points)/22)):
        points_tmp=points[j:j+22,:]
        points_tmp=points_tmp[np.argsort(points_tmp[:,0])]
        points_sort[j:j+22]=points_tmp
        j=j+22
    return points_sort,x

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

def fill_judge_left(points,x,img_ave,img_filling,black,white,img_debug=None):
    #lowは行index(20ピラーごと),colは列index(1start)
    low=0
    col=1
    col_flg=1
    #ピラー円中心を左上とした矩形内の充填判定
    for i in range(len(x)):

        if points[i,0]+41<=points[:,0].max()+2 and points[i,1]+41<=points[:,1].max()+2: #ピラー領域内の矩形のみを判定対象とする条件
            crop=img_ave[int(points[i,1]):int(points[i,1])+41,int(points[i,0]):int(points[i,0])+41] #矩形領域の取り出し

            # ★ デバッグ用：元画像（カラー）に矩形の外形線を描画
            if img_debug is not None:
                # 左上 (x0, y0) から 41×41 → 右下は (x0+40, y0+40)
                cv2.rectangle(
                    img_debug,
                    (int(points[i,0]), int(points[i,1])),
                    (int(points[i,0]) + 41, int(points[i,1]) + 41),
                    (0, 0, 255),  # 赤線（BGR）
                )
            #矩形を反時計回りに45度回転
            center=(41//2,41//2)
            M=cv2.getRotationMatrix2D(center, 45, 1.0)
            rotated=cv2.warpAffine(crop, M, (41, 41))

            #矩形の各列においていずれかの行に輝度値0のpxがあるかを判定
            judge=np.zeros(41) 
            for j in range(41):
                if (rotated[:,j]<=0).any():
                    judge[j]=1  #trueの場合，judgeに1を格納する

            #充填しているかの判定
            if (judge==1).all():    
                connected=is_connected_black_path_left(crop) #ピラー円間が輝度値0のpxでつながっているか判定、bool型で返す
                if connected==True:  
                    img_filling[low,col]=150
                    black+=41**2
                else:
                    img_filling[low,col]=250
                    white+=41**2
            else:
                img_filling[low,col]=250
                white+=41**2
            col=col+2
        #開始インデックス更新
        if (i+1)%22==0:
            low=low+1
            if col_flg==1:
                col_flg=0
            else:
                col_flg=1
            col=col_flg
    return img_filling,black,white,img_debug

def fill_judge_right(points,x,img_ave,img_filling,black,white,img_debug):
    #lowは行index(20ピラーごと),colは列index(1start)
    low=0
    col=0
    col_flg=0
    #ピラー円中心を右上とした矩形内の充填判定
    for i in range(len(x)):
        if points[i,0]-41>=points[:,0].min()-2 and points[i,1]+41<=points[:,1].max()+2: #ピラー領域内の矩形のみを判定対象とする条件
            crop=img_ave[int(points[i,1]):int(points[i,1])+41,int(points[i,0])-41:int(points[i,0])] #矩形領域の取り出し
            # ★ デバッグ用：元画像（カラー）に矩形の外形線を描画
            if img_debug is not None:
                # 左上 (x0, y0) から 41×41 → 右下は (x0+40, y0+40)
                cv2.rectangle(
                    img_debug,
                    (int(points[i,0])-41, int(points[i,1])),
                    (int(points[i,0]), int(points[i,1]) + 41),
                    (255, 0, 0),  # 赤線（BGR）
                )
            #矩形を反時計回りに45度回転
            center=(41//2,41//2)
            M=cv2.getRotationMatrix2D(center, 135, 1.0)
            rotated=cv2.warpAffine(crop, M, (41, 41))

            #矩形の各列においていずれかの行に輝度値0のpxがあるかを判定
            true=np.zeros(41)
            for j in range(41):
                if (rotated[:,j]<=50).any():
                    true[j]=1   #trueの場合，judgeに1を格納する

            #judgeのすべての値が1なら，その矩形を黒く塗る      
            if (true==1).all():
                connected=is_connected_black_path_right(crop) #ピラー円間が輝度値0のpxでつながっているか判定、bool型で返す
                if connected==True:  
                    img_filling[low,col]=150
                    black+=41**2
                else:
                    img_filling[low,col]=250
                    white+=41**2
            else:
                img_filling[low,col]=250
                white+=41**2
            col=col+2
        #開始インデックス更新
        if (i+1)%22==0:
            low=low+1
            if col_flg==1:
                col_flg=0
            else:
                col_flg=1
            col=col_flg
    return img_filling,black,white,img_debug

def save_graph(t,e,filepath):
    #充填率の時間変化の散布図
    plt.scatter(t,e)
    plt.xlabel('t(min)')
    plt.ylabel('filing factor(-)')
    plt.title('t-filfac')
    plt.savefig(filepath+'/imgana/filing_factor/filing_factor.png')

    #csv化  
    forcsv=pd.DataFrame({"time[min]":t,"filing factor[-]":e})
    forcsv.to_csv(filepath+'/imgana/filing_factor/filing_factor.csv')

def save_dist(x,row,k,filepath):
    forcsv=pd.DataFrame({"place[-]":x,"filing_factor_dist[-]":row})
    forcsv.to_csv(filepath+f"/imgana/filing_factor/xdist/filing_dist{k}.csv")
    
def get_col_filling(img_row_filling):
    rows, cols = img_row_filling.shape
    cfill = np.zeros(cols)
    for j in range(cols):
        # 列 j の充填率 = 列 j の黒(150) の個数 ÷ 全行数
        cfill[j] = np.count_nonzero(img_row_filling[:, j] == 150) / rows

    return cfill

def makedir(filepath):
    os.makedirs(filepath+f"/imgana/filing_factor/xdist",exist_ok=True)

def main():
    m=50    #連続画像枚数
    n=10    #解析スタック数
    s=150    #2値化閾値（ピラー円と粒子の輝度値のみが0になる値に設定）
    t=range(0,10) #画像撮影時間(min)
    e=np.zeros(n)  #充填率格納箱
    filepath="F:/experiment/2025_12_20_grid_particle_8um halfcircle_15um"
    makedir(filepath)
    for k in range(0,10):
        img=tiff.imread(filepath+f"/{k}.tiff")

        # 画像サイズの取得
        height = img.shape[1]
        width = img.shape[2]

        #連続画像の平均を取り，流動粒子を排除
        img_ave=moving_flowparticle(img,height,width,m)

        cv2.imwrite(filepath+f"/imgana/filing_factor/ave{k}.tiff",img_ave)

        circles=hough_circle(img_ave)

        img_ave=two_value(img_ave,height,width,s)

        cv2.imwrite(filepath+f"/imgana/filing_factor/two_value{k}.tiff",img_ave)

        #2値化後の画像をグレースケールで読み込み
        img_ave=cv2.imread(filepath+f"/imgana/filing_factor/two_value{k}.tiff",cv2.IMREAD_GRAYSCALE)
        img_filling=np.zeros((42,43))#判定結果の可視化1px=1充填

        img_ave=circle_line(img_ave,circles,0)

        #cv2.imwrite(filepath+f"/imgana/filing_factor/add_circle_line{k}.tiff", img_ave)

        img_ave=circle_line(img_ave,circles,1)

        cv2.imwrite(filepath+f"/imgana/filing_factor/add_circle{k}.tiff", img_ave) #円周のみで十分か

        points,x=get_circlepx(circles)
    
        #充填率算出のための格納箱
        black=0
        white=0

        # ★ デバッグ用画像（矩形の外形線を描くキャンバス）
        img_debug = cv2.cvtColor(img_ave.copy(), cv2.COLOR_GRAY2BGR)

        img_filling,black,white,img_debug=fill_judge_left(points,x,img_ave,img_filling,black,white,img_debug)         
        img_filling,black,white,img_debug=fill_judge_right(points,x,img_ave,img_filling,black,white,img_debug)

        #充填領域を黒く塗った画像の保存
        cv2.imwrite(filepath+f"/imgana/filing_factor/result{k}.png",img_filling)

        # ★ 矩形のメッシュを描いた画像を保存
        cv2.imwrite(filepath+f"/imgana/filing_factor/rect_mesh{k}.png", img_debug)

        row_filing=get_col_filling(img_filling)
        x_point=range(43)

        save_dist(x_point,row_filing,k,filepath)

        #充填率の算出
        e[k]=black/(black+white)

    save_graph(t,e,filepath)

if __name__ == "__main__":
    main()