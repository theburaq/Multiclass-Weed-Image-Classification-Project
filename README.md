# 🌱 Multiclass Weed Image Classification Using Deep Learning 🌾
This is an AgriAI HIVE project done during my Internship with LiveAI.

## 🌟 Project Overview
Weed infestation is a critical challenge in sustainable agriculture, impacting crop yields and increasing reliance on herbicides. This project addresses the need for precise weed identification by leveraging deep learning to classify 8 distinct weed categories using over 17,000 RGB images. The project also features a user-friendly web application interface developed using Streamlit for real-time weed classification.

## 🔍 Objective
The goal is to develop a model that can help farmers automate the identification of various weed species, reducing manual effort and improving crop yield.

## 🚀 Features
* **Multiclass Classification:** Classifies 8 distinct weed categories.
* **Transfer Learning:** Utilized pretrained ResNet-50 and Inception-v3 architectures for experimentation.
* **Data Augmentation:** Applied augmentation techniques to address data imbalance.
* **Custom Model:** Built a CNN model from scratch for performance benchmarking.

 ## 📂 Dataset
* **Source:** RGB images of various plant weeds.
* **Format**: The data is stored in `.csv` format. 
* **Size:** Over 17,000 images.
* **Categories:** 8 distinct weed types.
* You can download the dataset:
    * From TesnsorFlow dataset hub: [https://www.tensorflow.org/datasets/catalog/deep_weeds](https://www.tensorflow.org/datasets/catalog/deep_weeds), OR 
    * From GitHub here- images & dataset: [https://github.com/AlexOlsen/DeepWeeds/tree/master](https://github.com/AlexOlsen/DeepWeeds/tree/master) ; labels: [https://github.com/AlexOlsen/DeepWeeds/tree/master/labels](https://github.com/AlexOlsen/DeepWeeds/tree/master/labels) .

**Preprocessing Steps:**
* Resizing images to a uniform input size.
* Normalization for consistency across the dataset.

## 🛠️ Tools & Technologies
* **Framework:** PyTorch
* **Models:** ResNet-50, Inception-v3, and a custom CNN
* **Front-End:** Streamlit
* **Programming Language:** Python
* **Additional Libraries:** NumPy, Pandas, Matplotlib, Torchvision

## 📊 Methodology
1. **Data Preprocessing:**
  * Data cleaning and augmentation techniques.
  * Splitting data into training, validation, and testing sets.
2. **Model Development:**
 * **Transfer Learning:** Fine-tuned ResNet-50 and Inception-v3 pretrained on ImageNet.
 * **Custom Model:** Designed a CNN(ResNet-50) from scratch for comparison.
3. **Training & Evaluation:**
 * Loss function: CrossEntropyLoss
 * Optimizer: Adam
 * Metrics: Confusion matrix & ROC/AUC curves

