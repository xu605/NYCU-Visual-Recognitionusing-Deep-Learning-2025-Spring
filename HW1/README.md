# **NYCU Visual Recognition using Deep Learning 2025 Spring HW1**  

**StudentID:** 110550133 
**Name:** 劉安之

## **Introduction**  
This project focuses on **image classification** using **ResNeXt50-32x4d**, a variant of ResNet that employs **grouped convolutions** to enhance feature extraction. We integrate advanced training techniques such as ** MixUp, CutMix, and label smoothing** to improve model performance.  

## **How to Install**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/xu605/NYCU-Visual-Recognitionusing-Deep-Learning-2025-Spring.git
   cd ./HW1
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**  
### **Training the Model and Test**  
Train the model using:  
```bash
python 110550133_HW1.py
```

## **Performance Snapshot**  
| Model | Test Accuracy |
|---------|--------------|
| ResNet50| **0.84** |
| ResNeXt50 | **0.86** |  
| ResNeXt50 + MixUp | **0.87** |
| ResNeXt50 + MixUp + label smoothing | **0.89** |
