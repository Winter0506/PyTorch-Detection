import os
import sys
import yaml
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
# 导入两个函数
from detection import detections, plot_save_result



conf_path = './conf/conf.yaml'
with open(conf_path, 'r', encoding='utf-8') as f:
    data = f.read()
cfg = yaml.load(data)

gtFolder = 'data/groundtruths'
detFolder = 'data/detections'
savePath = 'data/results'

# 四个参数各自的意义
results, classes = detections(cfg, gtFolder, detFolder, savePath)
plot_save_result(cfg, results, classes, savePath)
