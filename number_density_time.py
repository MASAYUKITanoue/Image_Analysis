import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import numpy as np
import cv2
#パラメータ設定
m=1  #撮影枚数
l=1.28*0.001  #1pxあたりの長さ（mm）
h=10*0.001    #流路高さ(mm)

#値格納箱
t=np.array((0,14))
w=np.zeros(2)
target_num=[0,14]

for idx,i in enumerate(target_num):
    #imagejからのデータの読み込み(1列目と2,3,4行目を消去後のデータ)
    stack_spots=pd.read_csv(f"F:/experiment/2025_08_18_grid_particle/imgana/numden/time/ave/ave_spots{i}.csv")

    stack_spots_cut_1=stack_spots[["POSITION_Y","POSITION_T"]]#POSITION_Y,POSITION_T列のみを取り出す

    stack_spots_cut_1.to_csv(f"F:/experiment/2025_08_18_grid_particle/imgana/numden/time/ave/output/allspots_cut{i}.csv")#切り取り後のcsvファイルの保存

    count=len(stack_spots_cut_1)

    img = cv2.imread(f"F:/experiment/2025_08_18_grid_particle/imgana/crop{i}.tiff", cv2.IMREAD_GRAYSCALE)
    
    height,width=img.shape

    w[idx]=count/(height*width*m*l*l*h)

    

#可視化
plt.scatter(t,w)
plt.xlabel('t(min)')
plt.ylabel('number density(/mm3)')
plt.title('t-num den')
plt.savefig('F:/experiment/2025_08_18_grid_particle/imgana/numden/time/ave/output/num_den_time.png')

#csv化
forcsv=pd.DataFrame({"time[min]":t,"numden[/mm3]":w})

forcsv.to_csv("F:/experiment/2025_08_18_grid_particle/imgana/numden/time/ave/output/num_den_time.csv")