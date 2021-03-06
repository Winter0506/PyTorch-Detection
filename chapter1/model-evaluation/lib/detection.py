import os
from Evaluator import *
import pdb

def getGTBoxes(cfg, GTFolder):

    files = os.listdir(GTFolder)   # 存入列表之中
    # print(files)
    files.sort() # 对列表进行排序
    # print(files)

    # classes类的列表  后面两个是字典
    classes = []  
    num_pos = {}
    gt_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        # print(nameOfImage)  将1.txt后面的txt去掉
        # print(f)
        fh1 = open(os.path.join(GTFolder, f), "r")
        # print(fh1) 对象
        
        for line in fh1:
            line = line.replace("\n", "")
            # print(type(line))
            if line.replace(' ', '') == '':  # 如果末尾有空行，跳出循环
                continue
            #　print(line)
            splitLine = line.split(" ")  # 将一行中内容存入splitLine列表中
            #　print(line)
            #　print(splitLine)

            cls = (splitLine[0])  # class
            left = float(splitLine[1])  # 后面四个数字转换成float 分别为左上右下
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])      
            one_box = [left, top, right, bottom, 0]  # 存入一个新的one_box之中，五个元素，最后一个是0
            # print(one_box)
              
            if cls not in classes:
                classes.append(cls)  # 在classes之中新增加一个元素，
                gt_boxes[cls] = {}  # 
                num_pos[cls] = 0

            num_pos[cls] += 1
            

            if nameOfImage not in gt_boxes[cls]:
                gt_boxes[cls][nameOfImage] = []
            # 添加进去
            gt_boxes[cls][nameOfImage].append(one_box)  

        #　print(classes)  # 列表包含各个种类 ['class1', 'class2']
        # print(gt_boxes)
        '''
        {'class1': {'1': [[14.0, 56.0, 50.0, 100.0, 0], 
                          [50.0, 90.0, 150.0, 189.0, 0], 
                          [458.0, 657.0, 580.0, 742.0, 0]]}, 
         'class2': {'1': [[345.0, 894.0, 432.0, 940.0, 0], 
                         [590.0, 354.0, 675.0, 420.0, 0]]}}
        '''
        # print(num_pos)　# {'class1': 3, 'class2': 2}  统计每个class的种类并记录数量   
        fh1.close()
    return gt_boxes, classes, num_pos

def getDetBoxes(cfg, DetFolder):

    files = os.listdir(DetFolder)
    files.sort()

    det_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(DetFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            score = float(splitLine[5])
            one_box = [left, top, right, bottom, score, nameOfImage]

            if cls not in det_boxes:
                det_boxes[cls]=[]
            det_boxes[cls].append(one_box)

        fh1.close()
        # print(det_boxes)
    return det_boxes

def detections(cfg,
               gtFolder,
               detFolder,
               savePath,
               show_process=True):
    
    # getGTBoxes函数  getDetBoxes函数  得到真实框 真实种类  和检测框
    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    det_boxes = getDetBoxes(cfg, detFolder)
    
    # 创建一个对象
    evaluator = Evaluator()

    # 返回内容为 
    return evaluator.GetPascalVOCMetrics(cfg, classes, gt_boxes, num_pos, det_boxes)

def plot_save_result(cfg, results, classes, savePath):
    
    
    plt.rcParams['savefig.dpi'] = 80
    plt.rcParams['figure.dpi'] = 130

    acc_AP = 0
    validClasses = 0
    fig_index = 0

    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index+=1
        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow=True)
        plt.grid()
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)


    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
