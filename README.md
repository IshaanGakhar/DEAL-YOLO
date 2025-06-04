# DEAL-YOLO: Drone-based Efficient Animal Localization using YOLO

[![arXiv](https://img.shields.io/badge/arXiv-2503.04698-b31b1b.svg)](https://arxiv.org/abs/2503.04698)

## Abstract

Although advances in deep learning and aerial surveillance technology are improving wildlife conservation efforts, complex and erratic environmental conditions still pose a problem, requiring innovative solutions for cost-effective small animal detection. This work introduces **DEAL-YOLO**, a novel approach that improves small object detection in Unmanned Aerial Vehicle (UAV) images by using multi-objective loss functions like Wise IoU (WIoU) and Normalized Wasserstein Distance (NWD), which prioritize pixels near the centre of the bounding box, ensuring smoother localization and reducing abrupt deviations. 

Additionally, the model is optimized through efficient feature extraction with Linear Deformable (LD) convolutions, enhancing accuracy while maintaining computational efficiency. The Scaled Sequence Feature Fusion (SSFF) module enhances object detection by effectively capturing inter-scale relationships, improving feature representation, and boosting metrics through optimized multiscale fusion. Comparison with baseline models reveals high efficacy with **up to 69.5% fewer parameters** compared to vanilla YOLOv8-N, highlighting the robustness of the proposed modifications.

Through this approach, our paper aims to facilitate the detection of endangered species, animal population analysis, habitat monitoring, biodiversity research, and various other applications that enrich wildlife conservation efforts. DEAL-YOLO employs a two-stage inference paradigm for object detection, refining selected regions to improve localization and confidence. This approach enhances performance, especially for small instances with low objectness scores.

## Key Features

- **Multi-objective Loss Functions**: Incorporates Wise IoU (WIoU) and Normalized Wasserstein Distance (NWD) for improved localization
- **Linear Deformable Convolutions**: Efficient feature extraction while maintaining computational efficiency
- **Scaled Sequence Feature Fusion (SSFF)**: Enhanced multiscale feature fusion for better object detection
- **Two-stage Inference**: Refined detection paradigm for improved performance on small objects
- **Parameter Efficiency**: Up to 69.5% fewer parameters compared to YOLOv8-N

## Model Architecture

![DEAL-YOLO Architecture](resources/Model_Diagram.pdf)

The architecture incorporates:
- **LDConv**: Linear Deformable Convolution layers
- **C2F**: Cross Stage Partial with 2 Convolutions
- **SPPF**: Spatial Pyramid Pooling - Fast
- **AddSSFF**: Scaled Sequence Feature Fusion module
- Multi-scale feature extraction at P2 (160×160), P3 (80×80), and P4 (40×40)

## Results

### Bucktales Dataset

| Model | #Params(M) | Precision | Recall | mAP₅₀ |
|-------|------------|-----------|--------|-------|
| YOLOv5-N [29] | 2.504 | 46.9 | 53.1 | 48.7 |
| YOLOv8-N [30] | 4.234 | 38.7 | 42.2 | 42.3 |
| YOLOv8-N [28] | 3.006 | 70.7 | 41.8 | 42.8 |
| YOLOv9-T [31] | 1.972 | 59.7 | 48.8 | 55.8 |
| YOLOv10-N [32] | 2.697 | 42.0 | 45.7 | 46.2 |
| Gold-YOLO [33] | 5.610 | 38.6 | 75.0 | 50.7 |
| RT-DETR [34] | 41.97 | 48.1 | 38.9 | 35.7 |
| Faster-RCNN [35] | 43060 | 63.6 | 75.2 | 59.7 |
| **DEAL-YOLO** | **0.994** | **75.3** | **58.2** | **48.5** |
| **DEAL-YOLO** | **0.994** | **85.3** | **87.8** | **47.6** |

### WAID Dataset

| Model | #Params(M) | Precision | Recall | mAP₅₀ |
|-------|------------|-----------|--------|-------|
| YOLOv8-T [36] | 6.068 | 93.7 | 92.3 | 94.23 |
| YOLOv5-S [29] | 7.200 | 96.9 | 92.9 | 96.3 |
| ADD-YOLO [37] | 1.500 | 93.0 | 91.0 | 95.0 |
| WILD-YOLO [38] | 12.380 | 92.8 | 91.41 | 95.0 |
| YOLOv8 [42] | 8.770 | 83.9 | 90.7 | 86.3 |
| MobileNet v2 [39] | 3.950 | 40.0 | 91.5 | 59.1 |
| RT-DETR [34] | 41.970 | 84.6 | 81.8 | 87.6 |
| YOLOv8 [28] | 3.010 | 88.5 | 85.4 | 89.7 |
| **DEAL-YOLO-LD** | **0.914** | **91.1** | **88.9** | **93.4** |
| **DEAL-YOLO** | **0.994** | **92.0** | **88.8** | **93.3** |
| **DEAL-YOLO-LD*** | **0.914** | **95.2** | **94.8** | **90.5** |
| | **0.994** | **95.9** | **95.3** | **90.8** |

**Note**: DEAL-YOLO consistently achieves competitive or superior performance while maintaining significantly fewer parameters, demonstrating excellent parameter efficiency for drone-based animal detection.

## Applications

- **Wildlife Conservation**: Detection of endangered species
- **Population Analysis**: Automated animal counting and monitoring
- **Habitat Monitoring**: Large-scale ecosystem surveillance
- **Biodiversity Research**: Species distribution and behavior studies
- **Conservation Planning**: Data-driven wildlife management decisions

## Citation

```bash
@misc{naidu2025dealyolodronebasedefficientanimal,
     title={DEAL-YOLO: Drone-based Efficient Animal Localization using YOLO}, 
     author={Aditya Prashant Naidu and Hem Gosalia and Ishaan Gakhar and Shaurya Singh Rathore and Krish Didwania and Ujjwal Verma},
     year={2025},
     eprint={2503.04698},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2503.04698}, 
}
