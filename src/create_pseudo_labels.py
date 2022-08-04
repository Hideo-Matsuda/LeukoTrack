from copy import deepcopy

######## about dataset ###############################
image_sets = ['0593','0598'] #seq name of test data
DET_PATH = './output/tracktor/' # directory path for tracking results
DETECTION_GT_PATH = './data/train/' # directory path for new gt(add pseudo labels)
TRACKTOR_GT_PATH = './data/biodata/test/' #directory path for source gt 
THRESH =  61*0.8 #(number of frame) * 0.8(MT)
IOU_THRESH = 0.5
H_LIMIT = 511
W_LIMIT = 511
#############################################

def calculate_iou(det, gt):
    det_x, det_y, det_w, det_h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    gt_x, gt_y, gt_w, gt_h = float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])

    x_lim = min(det_x+det_w, gt_x+gt_w)
    y_lim = min(det_y+det_h, gt_y+gt_h)

    if det_x > x_lim or gt_x > x_lim:
        return 0
    elif det_y > y_lim or gt_y > y_lim:
        return 0

    all_area = det_w*det_h + gt_w*gt_h
    double_w = x_lim - max(det_x, gt_x)
    double_h = y_lim - max(det_y, gt_y)
    double_area = double_w * double_h
    all_area -= double_area

    return double_area / all_area

def count_tracked(det_list, gt_list):
    tracked_num = 0
    for det in det_list:
        if calculate_iou(det[2:6], gt_list[int(det[0])-1][2:6]) >= IOU_THRESH:
            tracked_num += 1
    return tracked_num

def is_mostly_tracked(det_list, gt_list):
    tracked_num = 0
    for det in det_list:
        if calculate_iou(det[2:6], gt_list[int(det[0])-1][2:6]) >= IOU_THRESH:
            tracked_num += 1
    return tracked_num >= THRESH

def get_average_iou(det_list, gt_list):
    iou_sum = 0

    for det in det_list:
        iou_sum += calculate_iou(det[2:6], gt_list[int(det[0])-1][2:6])

    return iou_sum / len(det_list)

def create_gt(gt_list, num):
    with open(DETECTION_GT_PATH+num+'/gt/gt.txt', mode='w') as file:
        file.writelines(gt_list)

def create_gt_1(gt_list, num): # rotate by 90
    new_gt_list = []
    for gt in gt_list:
        gt = gt.split(',')
        new_gt = gt[0]+','+gt[1]+','
        x = H_LIMIT - int(gt[3]) - int(gt[5])
        y = gt[2]
        w = gt[5]
        h = gt[4]
        new_gt += str(x)+','+y+','+w+','+h+',1,1,1\n'
        new_gt_list.append(new_gt)
    with open(DETECTION_GT_PATH+num+'_1/gt/gt.txt', mode='w') as file:
        file.writelines(new_gt_list)

def create_gt_2(gt_list, num):# rotate by 180
    new_gt_list = []
    for gt in gt_list:
        gt = gt.split(',')
        new_gt = gt[0]+','+gt[1]+','
        x = W_LIMIT - int(gt[2]) - int(gt[4])
        y = H_LIMIT - int(gt[3]) - int(gt[5])
        w = gt[4]
        h = gt[5]
        new_gt += str(x)+','+str(y)+','+w+','+h+',1,1,1\n'
        new_gt_list.append(new_gt)
    with open(DETECTION_GT_PATH+num+'_2/gt/gt.txt', mode='w') as file:
        file.writelines(new_gt_list)

def create_gt_3(gt_list, num): #rotate by 270
    new_gt_list = []
    for gt in gt_list:
        gt = gt.split(',')
        new_gt = gt[0]+','+gt[1]+','
        x = gt[3]
        y = W_LIMIT - int(gt[2]) - int(gt[4])
        w = gt[5]
        h = gt[4]
        new_gt += x+','+str(y)+','+w+','+h+',1,1,1\n'
        new_gt_list.append(new_gt)
    with open(DETECTION_GT_PATH+num+'_3/gt/gt.txt', mode='w') as file:
        file.writelines(new_gt_list)


def main():
    for num in image_sets:
        print(num)
        with open(DET_PATH+num+'', mode='r') as file:
            lines = file.readlines()
        det_dict = {}
        for line in lines:
            line = line.split(',')
            if not line[1] in det_dict.keys():
                det_dict[line[1]] = []
            det_dict[line[1]].append(line)

        with open(TRACKTOR_GT_PATH+num+'/gt_non_evaluate/gt.txt', mode='r') as file:
            lines = file.readlines()
            gt_list = deepcopy(lines)
        gt_dict = {}
        for line in lines:
            line = line.split(',')
            if not line[1] in gt_dict.keys():
                gt_dict[line[1]] = []
            gt_dict[line[1]].append(line)

        gt_id_list = [str(i) for i in range(1,11)]
        n = 0
        for key, value in det_dict.items():
            for i in range(len(gt_id_list)):
                gt_id = gt_id_list[i]
                tracked_num = 0
        
        new_gt_list = []
        i = 10
        tracked_num = 0
        for det in det_dict.values():
            if len(det) < THRESH:
                continue
            gt_flag = False
            for gt in gt_dict.values():
                if is_mostly_tracked(det, gt):
                    gt_flag = True
                    tracked_num += 1
                    break
            if gt_flag:
                continue
            i += 1
            for temp in det:
                temp[0] = temp[0] + ','
                temp[1] = str(i) + ','
                temp[2] = str(int(float(temp[2]))) + ','
                temp[3] = str(int(float(temp[3]))) + ','
                temp[4] = str(int(float(temp[4]))) + ','
                temp[5] = str(int(float(temp[5]))) + ','
                temp[6] = '1,1,1\n'
                new_gt = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]
                new_gt_list.append(new_gt)
        new_gt_list = gt_list + new_gt_list
        print('mostly tracked : {}'.format(tracked_num))
        print('new gt num : {}'.format(new_gt_list[-1].split(',')[1]))
        create_gt(new_gt_list,num)
        create_gt_1(new_gt_list,num)
        create_gt_2(new_gt_list,num)
        create_gt_3(new_gt_list,num)


if __name__ == '__main__':
    main()
