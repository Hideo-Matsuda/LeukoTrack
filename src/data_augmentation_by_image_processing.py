import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt

######### about dataset ###########
image_name = 'MOT20-01'
group = 'MOT20'
image_dir = os.path.join('./data/train/',group, image_name,'img1')
output_dir1 = os.path.join('data/train/',group,image_name+'_1','img1')
output_dir2 = os.path.join('data/train/',group, image_name+'_2','img1')
output_dir3 = os.path.join('data/train/',group, image_name+'_3','img1')
W_LIMIT = 1920 #MOT20-01, MOT20-02: 1920, MOT20-03: 1173, MOT20-05: 1645
H_LIMIT = 1080 #MOT20-01, MOT20-02: 1080, MOT20-03: 880, MOT20-05: 1080

## For debug (draw bounding box)
debug = False
type = 1 #1:rotete by 90, 2:rotate by 180, 3:rotate by 270

#######################################

image_name_list = sorted(os.listdir(image_dir))
os.mkdir(output_dir1)
os.mkdir(output_dir2)
os.mkdir(output_dir3)

for seq in image_name_list:
    #print(seq)
    image = cv2.imread(os.path.join(image_dir,seq))
    img_1=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_dir1,seq), img_1)

    img_2=cv2.rotate(image,cv2.ROTATE_180)
    cv2.imwrite(os.path.join(output_dir2,seq), img_2)

    img_3=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(output_dir3,seq), img_3)

df = pd.read_csv(os.path.join('./data/train',group, image_name,'gt/gt.txt'),header =None,names =['t','id','x','y','w','h','detection','class_no', 'visibility'])
bb_list_90, bb_list_180, bb_list_270 = [],[],[]

for row in df.itertuples():
    #print(row)
    t = row.t
    id = row.id
    x = row.x
    y = row.y
    w = row.w
    h = row.h
    d = row.detection
    c = row.class_no
    v = row.visibility

    x_180= W_LIMIT-x-w
    y_180 = H_LIMIT-y-h
    w_180 = w
    h_180 = h

    bb_180 = [t,id,x_180,y_180,w_180,h_180,d,c,v]
    bb_list_180.append(bb_180)

    x_90 = H_LIMIT-y-h
    y_90 = x
    w_90 = h
    h_90 = w

    bb_90 = [t,id,x_90,y_90,w_90,h_90,d,c,v]
    bb_list_90.append(bb_90)

    x_270= y
    y_270 = W_LIMIT-x-w
    w_270 = h
    h_270 = w

    bb_270 = [t,id,x_270,y_270,w_270,h_270,d,c,v]
    bb_list_270.append(bb_270)


df_90 = pd.DataFrame(bb_list_90,columns = ['t','id','x','y','w','h','detection','class_no', 'visibility'])
df_90.to_csv(os.path.join('data/train',group, image_name+'_1','gt.txt'),header=None,index=None)

df_180 = pd.DataFrame(bb_list_180,columns = ['t','id','x','y','w','h','detection','class_no', 'visibility'])
df_180.to_csv(os.path.join('data/train',group, image_name+'_2','gt.txt'),header=None,index=None)

df_270 = pd.DataFrame(bb_list_270,columns = ['t','id','x','y','w','h','detection','class_no', 'visibility'])
df_270.to_csv(os.path.join('data/train',group, image_name+'_3','gt.txt'),header=None,index=None)

# draw bounding box for debug
if debug:
    img_dir = os.path.join('./data/train/',group,image_name+'_'+type,'img1')
    img_name_list= sorted(os.listdir(img_dir))
    img_list = [cv2.imread(os.path.join(img_dir,s)) for s in img_name_list]

    num_tracklets = df_90['id'].max()+1
    color_list = []
    for _ in range(num_tracklets):
        color_list.append(np.random.randint(0,256,(3,),dtype=np.uint8))

    for row in df_90.itertuples():
        img_no = int(row.t)
        bbox_x = int(row.x)
        bbox_y = int(row.y)
        bbox_w = int(row.w)
        bbox_h = int(row.h)
        tracklet_id = int(row.id)
        color = tuple(map(int, color_list[tracklet_id]))

        cv2.rectangle(img_list[img_no-1], (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, 2)
        cv2.putText(img_list[img_no-1], str(tracklet_id), (bbox_x, bbox_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

    os.makedirs(os.path.join('data/train',group,image_name+'_'+type), exist_ok=True)
    for img_name, img in zip(img_name_list, img_list):
        cv2.imwrite(os.path.join('data/train',group,image_name+'_'+type+'_debug',img_name), img)
