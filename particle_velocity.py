import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

#パラメータ設定
l=1.28 #1pxあたりの長さ
n=75 #framerate
filepath="F:/experiment/2025_08_18_grid_particle"

#csv整理と線形性による排除
stack_tracks=pd.read_csv(filepath+"/imgana/velo/tracks0.csv")
stack_spots=pd.read_csv(filepath+"/imgana/velo/spots0.csv")
stack_tracks_cut=stack_tracks[["TRACK_ID","TRACK_Y_LOCATION","TRACK_DURATION","TRACK_MEAN_SPEED","LINEARITY_OF_FORWARD_PROGRESSION"]]
stack_spots_cut=stack_spots[["TRACK_ID","POSITION_X","POSITION_Y","POSITION_T"]]
stack_merge=pd.merge(stack_tracks_cut,stack_spots_cut,on="TRACK_ID",how="left")
stack_merge.to_csv(filepath+"/imgana/velo/output/Stack_merge.csv")
stack_cut=stack_merge[(0.99<=stack_merge["LINEARITY_OF_FORWARD_PROGRESSION"])&(stack_merge["LINEARITY_OF_FORWARD_PROGRESSION"]<=1)]
stack_cut.to_csv(filepath+"/imgana/velo/output/Stack_cut.csv")
track_num=stack_cut["TRACK_ID"].unique()

#値格納箱
result_x=np.zeros((len(track_num),2))
result_y=np.zeros((len(track_num),2))

#粒子速度分布計算
for idx,i in enumerate(track_num):
    track_cut=stack_cut[stack_cut["TRACK_ID"]==i]
    track_xmax=track_cut.loc[track_cut["POSITION_T"].idxmax(),"POSITION_X"]
    track_xmin=track_cut.loc[track_cut["POSITION_T"].idxmin(),"POSITION_X"]
    track_ymax=track_cut.loc[track_cut["POSITION_T"].idxmax(),"POSITION_Y"]
    track_ymin=track_cut.loc[track_cut["POSITION_T"].idxmin(),"POSITION_Y"]
    result_x[idx,0]=track_cut["TRACK_Y_LOCATION"].max()*l
    result_y[idx,0]=track_cut["TRACK_Y_LOCATION"].max()*l
    result_x[idx,1]=(track_xmax-track_xmin)*l*n/track_cut["TRACK_DURATION"].max()
    result_y[idx,1]=(track_ymax-track_ymin)*l*n/track_cut["TRACK_DURATION"].max()

result_x=pd.DataFrame(result_x,columns=["y_location(um)","xaxis_velocity(um/s)"])
result_y=pd.DataFrame(result_y,columns=["y_location(um)","yaxis_velocity(um/s)"])
result_x.to_csv(filepath+"/imgana/velo/output/xaxis_velocity.csv")
result_y.to_csv(filepath+"/imgana/velo/output/yaxis_velocity.csv")

plt.figure()
plt.scatter(result_x["y_location(um)"], result_x["xaxis_velocity(um/s)"])
plt.xlabel('y_location(um)')
plt.ylabel('xaxis_velocity(um/s)')
plt.title('velocity')
plt.savefig(filepath+'/imgana/velo/output/xaxis_velocity.png')

plt.figure()
plt.scatter(result_y["y_location(um)"], result_y["yaxis_velocity(um/s)"])
plt.xlabel('y_location(um)')
plt.ylabel('yaxis_velocity(um/s)')
plt.title('velocity')
plt.savefig(filepath+'/imgana/velo/output/yaxis_velocity.png')

