## Camera Radar Fusion 

While recent autonomous systems often rely on cameras (looking at you, Tesla), radar–camera fusion remains a powerful and underexplored approach. Cameras provide rich semantics but fail in poor weather; radars offer range and velocity robustness but sparse data. Fusing both yields stronger perception.

Developing an effective radar–camera fusion system requires key design choices — **where** and **how** to align the modalities, and **when** and **how** to combine their features for optimal performance.

---

## What This Repository Achieves  

This repository explores these design aspects through practical implementations of radar–camera fusion for **2D object detection**. It provides:

- **Training pipelines** for camera-only, early-fusion, and middle-fusion models  
- **Explicit feature alignment** using cross-attention to enhance radar–camera interaction  
- **Projection utilities** for transforming radar data into image or BEV representations  
- **End-to-end training and evaluation scripts** implemented in PyTorch  

---

For a detailed walkthrough, see the included Jupyter notebook with examples and visualizations:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l6XTc8O4Pp661EwiU5N930gfc3Vp9G_D#scrollTo=GALZjjMCK8ea)
