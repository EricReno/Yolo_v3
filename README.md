# Yolo_V3 (You only look once)
yolo_v3的pytorch源码，数据集为VOC20，包含训练、推理、验证、部署脚本

- 传送门：

https://arxiv.org/abs/1506.02640 (v1)
https://arxiv.org/abs/1612.08242 (v2)
https://arxiv.org/abs/1804.02767 (v3)

## 数据集: VOC 
- **test**: (VOC2007, test) : 4952
- **train**: (VOC2007, trainval), (VOC2012, trainval) : 16553
- **CLASSES_NAMES**: 

|             |          |         |           |           |
| :---------: | :------: | :-----: | :-------: | :-------: |
|  aeroplane  | bicycle  |  bird   |   boat    | bottle    |
|     bus     |   car    |  cat    |  chair    | cow       |
| diningtable |   dog    | horse   | motorbike | person    |
| pottedplant |  sheep   |  sofa   |  train    | tvmonitor |

- **官方网址** 

    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## 通用设置
| Size  |  BS | Pretrain| Epoch| Obj | Cls | Box | NMS | Confidence| APT | Learningrate|
| :---: |:---:|  :---:  | :---:|   :---: |:---:  | :---:      | :---:    | :---:    | :---:    |:---:    |
|512x512|  64 |   CoCo  |  160 |   1.0 | 1.0  | 5.0        | 0.5      |  0.3     | SGD| 0.01|

|DataAugmentation|
|   :---:     |
|RandomSaturationHue|
|RandomContrast|
|RandomBrightness|
|RandomSampleCrop|
|RandomExpand|
|RandomHorizontalFlip|

## Results:
| TAG  |  Size|    mAP    |    GFLOPs     |Params |Pt_Size| FPS |
| :---: |   :---:   | :---:   |  :---:  |:---:  |:---:  |:---:  |
|v3_tiny|   512   |56.17%  |   5.18      | 2.43| 19M|75.44(1050Ti)|
|v3_Darknet53|   512   |75.71%  |  133.40      | 57.43| 442M|10.32(1050Ti)|