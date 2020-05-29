import os
import sys
from collections import Counter
import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

class Evaluator:
    # 获取参数
    def GetPascalVOCMetrics(self,
                            cfg,
                            classes, 
                            gt_boxes,
                            num_pos,
                            det_boxes):

        # ret groundTruths detections三个空列表
        ret = []
        groundTruths = []
        detections = []
        
        for c in classes:
            # ['class1', 'class2']
            # 通过类别作为关键字，得到每个类别的预测，标签及总标签数
            dects = det_boxes[c]  
            """
            第一类预测 [[12.0, 58.0, 53.0, 96.0, 0.87, '1'], 
                       [51.0, 88.0, 152.0, 191.0, 0.98, '1'], 
                       [243.0, 546.0, 298.0, 583.0, 0.83, '1']]
            """
            # print(dects)
            gt_class = gt_boxes[c]
            # print(gt_class)
            """
            {'1': [[14.0, 56.0, 50.0, 100.0, 0], [50.0, 90.0, 150.0, 189.0, 0], [458.0, 657.0, 580.0, 742.0, 0]]}
            """
            npos = num_pos[c]
            # print(npos) # 3
            # 利用得分作为关键字，对预测框按照得分从高到低排序
            dects = sorted(dects, key=lambda conf: conf[4], reverse=True)   # 第四个参数就是得分
            # print(dects)
            # print(dects)
            # 设置两个与预测边框长度相同的列表，标记是True Positive还是False Positive
            TP = np.zeros(len(dects))
            # print(TP)  创建与dects长度相同的数组
            FP = np.zeros(len(dects))
            # 对某一个类别的预测框进行预测 
            for d in range(len(dects)):
                # 将IOU默认值设置为最低  接近于0了
                iouMax = sys.float_info.min
                # print(iouMax)
                # 遍历与预测框同一图像中的同一类别的标签，计算IoU
                if dects[d][-1] in gt_class:
                    # print(len(gt_class[dects[d][-1]]))
                    for j in range(len(gt_class[dects[d][-1]])):
                        # 调用静态方法
                        iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                        if iou > iouMax:
                            iouMax = iou
                            jmax = j  # 记录与预测有最大IoU的标签

                    # 如果最大Iou大于预支，并且没有被匹配过，赋予TP
                    if iouMax >= cfg['iouThreshold']:
                        if gt_class[dects[d][-1]][jmax][4] == 0:
                            TP[d] = 1
                            gt_class[dects[d][-1]][jmax][4] == 1  # 标记为匹配过
                        # 如果被匹配过，赋予FP
                        else:
                            FP[d] = 1
                    # 如果最大IoU没有超过阈值，赋予FP
                    else:
                        FP[d] = 1
                # 如果对应图像中没有该类别的标签，赋予FP
                else:
                    FP[d] = 1
            # 利用cumsum()函数，计算累计的FP与TP
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            # 得到每个点的recall
            rec = acc_TP / npos
            # 得到每个点的precision
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            print(' ')
            # 进一步计算得到AP
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            print([ap, mpre, mrec, ii])
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
            print(r)
        return ret, classes

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[i+1] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        # ap的计算
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False: # 两个框不重合，直接返回0
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB) #计算得到交叉面积
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union  # 计算iou
        if iou < 0:
            import pdb
            pdb.set_trace()
        assert iou >= 0  # 断言iou肯定大于0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    # 两个框不相重合的情况  返回False
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        # 计算重合部分的上 左 下 右四个边的值，注意最大最小函数的使用
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        # 返回重合部分面积
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)  # 两个框的总面积

    @staticmethod
    # 计算每个单个框的面积
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
