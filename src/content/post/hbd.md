---
title: Usage of the Hong Kong University Architectural Dataset
description: The Hong Kong University Architectural Dataset is a project of my advisor, used for training models for architectural component segmentation.
publishDate: 16 Sep 2024
tags:
  - Machine
  - Learning
---
## Basic Information
In image-driven 3D building reconstruction, instance segmentation is fundamental to pixel-wise building component detection, which can be fused with 3D data like point clouds and meshes via camera projection for semantic reconstruction. While deep learning-based segmentation has obtained promising results, it relies heavily on large-scale datasets for training. Unfortunately, existing large-scale image datasets often include irrelevant objects that obstruct building components, making them unsuitable for 3D building reconstruction. This paper addresses this gap by introducing a large-scale building image dataset to facilitate building component segmentation for 3D reconstruction. The dataset comprises 3378 images captured from both interiors and exteriors of 36 university buildings, annotated with 49,380 object instances across 11 classes. Rigorous quality control measures were employed during data collection and annotation. Evaluation of five typical deep learning-based instance segmentation models demonstrates the dataset’s suitability for training and its value as a benchmark dataset for building component segmentation.  
Below is the link to the paper：
*Mun On Wong, Huaquan Ying, Mengtian Yin, Xiaoyue Yi, Lizhao Xiao, Weilun Duan, Chenchen He, Llewellyn Tang,*
*Semantic 3D reconstruction-oriented image dataset for building component segmentation,Automation in Construction,Volume 165,2024,105558,ISSN 0926-5805,*
https://doi.org/10.1016/j.autcon.2024.105558.
https://www.sciencedirect.com/science/article/pii/S0926580524002942
## My Work
I need to convert this dataset into the YOLOV8 format and then train it for architectural component segmentation. This segmentation can be used in point cloud projection to project individual components, such as walls, windows, or ceilings, where I only need specific elements like walls, windows, or ceilings.  
[Download Hong Kong University Architectural Dataset in YOLOV8 Format](https://huggingface.co/datasets/ColamanAI/3DPointCloud/tree/main/3D%E7%82%B9%E4%BA%91%E9%87%8D%E5%BB%BA/%E6%95%B0%E6%8D%AE%E9%9B%86/%E9%A6%99%E6%B8%AF%E5%A4%A7%E5%AD%A6)
