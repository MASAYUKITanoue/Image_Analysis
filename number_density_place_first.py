import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import numpy as np

#パラメータ設定
n=30    #y軸の分割個数
x=1695   #検査体積のx軸方向のpx数
y=1776/n   #検査体積のy軸方向のpx数
m=50    #撮影枚数
l=1.28*0.001  #1pxあたりの長さ（mm）
t=10*0.001    #流路高さ(mm)
filepath="F:/experiment/2025_08_18_grid_particle"

#imagejからのデータの読み込み(1列目と2,3,4行目を消去後のデータ)
stack_spots=pd.read_csv(filepath+"/imgana/numden/allspots0.csv")

stack_spots_cut_1=stack_spots[["POSITION_Y","POSITION_T"]]#POSITION_Y,POSITION_T列のみを取り出す

stack_spots_cut_1.to_csv(filepath+"/imgana/numden/place/spots_cut.csv")#切り取り後のcsvファイルの保存

#データ格納場所の作成
a=np.zeros(n)
b=np.zeros(n)

#数密度の計算（y軸の分割領域ごとにcsvを分ける→行数をその領域における粒子の個数とし変数bに格納）
for i in range(n):
    result=stack_spots_cut_1[(i*y<=stack_spots_cut_1["POSITION_Y"])&(stack_spots_cut_1["POSITION_Y"]<=(i+1)*y)]
    result.to_csv(filepath+f"/imgana/numden/place/results_{i}.csv",index=False)
    data=pd.read_csv(filepath+f"/imgana/numden/place/results_{i}.csv")
    a[i]=((i*y+(i+1)*y)*l*1000)/2
    b[i]=len(data)

print(b)
num=b/(y*x*m*l*l*t)

print(num)#値の確認

#可視化
plt.scatter(a, num)
plt.xlabel('y(um)')
plt.ylabel('number density(/mm3)')
plt.title('y-num den')
plt.savefig(filepath+'/imgana/numden/place/num_den_place.png')

#csv化
forcsv=pd.DataFrame({"Position:y[um]":a,"numden[/mm3]":num})

forcsv.to_csv(filepath+"/imgana/numden/place/num_den_place.csv")