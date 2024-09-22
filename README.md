# Yolo_V3 (You only look once)

- yolo_v3的pytorch源码，含训练、推理、评测、部署脚本，论文传送门：

https://arxiv.org/abs/1506.02640 (v1)

https://arxiv.org/abs/1612.08242 (v2)

https://arxiv.org/abs/1804.02767 (v3)

## 数据集: VOC 
- **test**: (VOC2007, test) : 4952
- **train**: (VOC2007, trainval), (VOC2012, trainval) : 16553
- **CLASSES_NAMES**:

|             |          |         |           |           |
| :---------: | :------: | :-----: | :-------: | :-------: |
|  aeroplane  | bicycle  |  bird   |   boat    | bottle    |
|     bus     |   car    |  cat    |  chair    | cow       |
| diningtable |   dog    | horse   | motorbike | person    |
| pottedplant |  sheep   |  sofa   |  train    | tvmonitor |

- **官方网址** 

    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## 通用设置
|DataAugmentation    |
|:---:               |
|RandomSaturationHue |
|RandomContrast      |
|RandomBrightness    |
|RandomSampleCrop    |
|RandomExpand        |
|RandomHorizontalFlip|

|BS   |Pretrained|Epoch|NMS_TH.|Confidence_TH.|Optimizer|LearningRate|LrSheduler|
|:---:|:---:     |:---:|:---: |:---:          |:---:    |:---:       |:---:     |
|  64 |CoCo      |160  |0.5   |0.001          |SGD      |0.01        |linear    |

## Results:
|TAG              |Size |mAP   |GFLOPs|Params |*.pt |FPS-3060|
|:---:            |:---:|:---: |:---: |:---:  |:---:|:---:   |
|Yolo_v3_Tiny     |512  |69.52%|  4.56|  2.39M|19.3M| 75.1001|
|Yolo_v3_Darknet53|512  |78.68%|108.62| 56.89M| 442M|  5.0489|

<table>
<tr><th>Yolo_v3_Tiny</th> <th>Yolo_v3_Darknet53</th></tr>
<tr>
<td>
    
|ClassNames |AP   |
|--         |--   |
|aeroplane  |0.752|
|bicycle    |0.783|
|bird       |0.700|
|boat       |0.601|
|bottle     |0.514|
|bus        |0.782|
|car        |0.799|
|cat        |0.808|
|chair      |0.525|
|cow        |0.756|
|diningtable|0.610|
|dog        |0.742|
|horse      |0.801|
|motorbike  |0.766|
|person     |0.753|
|pottedplant|0.426|
|sheep      |0.686|
|sofa       |0.654|
|train      |0.782|
|tvmonitor  |0.662|
|mAP        |0.695|

</td>
<td>
    
|ClassNames |AP   |
|--         |--   |
|aeroplane  |0.815|
|bicycle    |0.855|
|bird       |0.827|
|boat       |0.698|
|bottle     |0.640|
|bus        |0.864|
|car        |0.855|
|cat        |0.875|
|chair      |0.639|
|cow        |0.872|
|diningtable|0.699|
|dog        |0.858|
|horse      |0.874|
|motorbike  |0.852|
|person     |0.809|
|pottedplant|0.535|
|sheep      |0.790|
|sofa       |0.736|
|train      |0.858|
|tvmonitor  |0.781|
|mAP        |0.787|

</td>
</tr> 
</table>

<video src="https://github.com/user-attachments/assets/d5811825-8c58-4f0f-9067-a79d0c9966dc" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>
