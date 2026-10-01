import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import numpy as np
import cv2

def read_calc(target_num,filepath,w,l,h,m):
    for idx,i in enumerate(target_num):
        #imagejからのデータの読み込み(1列目と2,3,4行目を消去後のデータ)
        stack_spots=pd.read_csv(filepath+f"/imgana/numden/time/ave/ave_spots{i}.csv")

        stack_spots_cut_1=stack_spots[["POSITION_Y","POSITION_T"]]#POSITION_Y,POSITION_T列のみを取り出す

        stack_spots_cut_1.to_csv(filepath+f"/imgana/numden/time/ave/output/allspots_cut{i}.csv")#切り取り後のcsvファイルの保存

        count=len(stack_spots_cut_1)

        img = cv2.imread(filepath+f"/imgana/crop{i}.tiff", cv2.IMREAD_GRAYSCALE)
        
        height,width=img.shape

        w[idx]=count/(height*width*m*l*l*h)
    return w
    
def output_data(filepath,t,w):
    #可視化
    plt.scatter(t,w)
    plt.xlabel('t(min)')
    plt.ylabel('number density(/mm3)')
    plt.title('t-num den')
    plt.savefig(filepath+'/imgana/numden/time/ave/output/num_den_time.png')

    #csv化
    forcsv=pd.DataFrame({"time[min]":t,"numden[/mm3]":w})
    forcsv.to_csv(filepath+"/imgana/numden/time/ave/output/num_den_time.csv")

def main():
    #パラメータ設定
    m=50  #連続画像枚数
    l=1.28*0.001  #1pxあたりの長さ（mm）
    h=10*0.001    #流路高さ(mm)
    filepath="F:/experiment/2025_08_18_grid_particle"
    t=np.array((0)) #撮影時刻[min]
    target_num=[0,14]   #画像およびcsvファイルの番号（tに代入した時刻に対応する番号を入力）
    #値格納箱
    w=np.zeros(1)  #数密度格納用配列(配列サイズ設定の必要有)
    
    #csv読み込みと数密度計算
    w=read_calc(target_num,filepath,w,l,h,m)

    #データの出力
    output_data(filepath,t,w)

if __name__ == "__main__":
    main()