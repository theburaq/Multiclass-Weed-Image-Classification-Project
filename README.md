# 🌱 Multiclass Weed Image Classification Using Deep Learning 🌾
This is an AgriAI HIVE project done during my Internship with LiveAI.

## 🌟 Project Overview
Weed infestation is a critical challenge in sustainable agriculture, impacting crop yields and increasing reliance on herbicides. This project addresses the need for precise weed identification by leveraging deep learning to classify 8 distinct weed categories using over 17,000 RGB images. The project also features a user-friendly web application interface developed using Streamlit for real-time weed classification.

## 🔍 Objective
The goal is to develop a model that can help farmers automate the identification of various weed species, reducing manual effort and improving crop yield. We experimented on weed datasets by DeepWeed using two CNN architectures namely ResNet-50 and Inception-v3, and compare the results.

## 💡 Features
* **Multiclass Classification:** Classifies 8 distinct weed categories.
* **Data Augmentation:** Applied augmentation techniques to address data imbalance.
* **Transfer Learning:** Utilized a pretrained Inception-v3 for performance benchmarking. 
* **Custom Model:** Built a CNN (ResNet-50) model from scratch architectures for experimentation.
* **Streamlit Integration:** Real-time predictions through a web-based application.

 ## 📂 Dataset
* **Source:** RGB images of various plant weeds.
* **Format**: The data is stored in `.csv` format. 
* **Size:** Over 17,000 images.
* **Categories:** 8 distinct weed types.
* You can download the dataset:
    * From TesnsorFlow dataset hub: [https://www.tensorflow.org/datasets/catalog/deep_weeds](https://www.tensorflow.org/datasets/catalog/deep_weeds), OR 
    * From GitHub here- images & dataset: [https://github.com/AlexOlsen/DeepWeeds/tree/master](https://github.com/AlexOlsen/DeepWeeds/tree/master) ; labels: [https://github.com/AlexOlsen/DeepWeeds/tree/master/labels](https://github.com/AlexOlsen/DeepWeeds/tree/master/labels) 

## 🛠️ Tools & Technologies
* **Framework:** PyTorch
* **Models:** A pre-trained Inception-v3 and Custom ResNet-50 
* **Front-End:** Streamlit
* **Programming Language:** Python
* **Additional Libraries:** NumPy, Pandas, Matplotlib, Torchvision

## 📊 Methodology
1. **Data Preprocessing:**
  * Data cleaning and augmentation techniques.
  * Resizing images to a uniform input size.
  * Normalization for consistency across the dataset.
2. **Model Development:**
  * **Transfer Learning:** Inception-v3 pretrained on ImageNet. 
    * Pre-trained Inception-v3 model file: [Pre-trained Inception-v3](/4_Inception-v3-pre-trained-model_Final.ipynb)
  * **Custom Model:** Designed a customed ResNet-50 model from scratch & finetuned it for comparison.
    * ResNet-50 'custom model' file: [ResNet-50 from scratch](/1_ResNet50_from_scratch_Final.ipynb)
    * ResNet-50 'custom model with data augmentation' file: [ResNet-50 from scratch with data augmentation](/2_ResNet50-from-scratch-data-augmentation_Final.ipynb)
    * ResNet-50 'custom model with pre-trained weights' file: [ResNet-50 from scratch with pre-trained weights](/3_ResNet50-from-scratch-pre-trained-weights_Final.ipynb)

3. **Training & Evaluation:**
 * Loss function: CrossEntropyLoss
 * Optimizer: Adam
 * Metrics: Confusion matrix & ROC/AUC curves
   
4. We've also experimented by applying a **Ray Tune** to find out the best hyperparameters: [Ray Tune](/5_ray_tuner_final.ipynb)
   
5. **Deployment:**
 * Developed a web interface using Streamlit for real-time predictions.

## 📈 Key Results
* Best Performing Model:
  * Both ResNet-50 (custom with pre-trained weights) & Inception-v3 (pre-trained) models have had training accuracy of nearly 100% but differed by little margin in validation accuracy.
  * Validation Accuracy for:
     * ResNet-50 (custom with pre-trained weights)- around 90%
     * Inception-v3 (pre-trained)- around 95% 

## 🎥 Demo
Click here, [Weed Image Classification Demo](/Streamlit_Agri_Project.mp4) , to watch the demo video of the web application.
* Features in the demo:
  * Upload an image of a weed.
  * Real-time classification display.
  * Simple and intuitive user interface.

## 📘 References
* The Research paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/)
* [https://paperswithcode.com/paper/deepweeds-a-multiclass-weed-species-image/review/](https://paperswithcode.com/paper/deepweeds-a-multiclass-weed-species-image/review/)

## 🚀 Future Improvements
* Extend the dataset to include additional weed species.
* Explore advanced architectures like Vision Transformers.
* Deploy the Streamlit app on a cloud platform for wider accessibility.
* Incorporate explainability tools like Grad-CAM for better interpretability.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


