# 🌱 Multiclass Weed Image Classification Using Deep Learning 🌾
This is an AgriAI HIVE project done during my Internship with [LiveAI](https://www.liveai.eu/) .

## 🌟 Project Overview
Weed infestation is a critical challenge in sustainable agriculture, impacting crop yields and increasing reliance on herbicides. This project addresses the need for precise weed identification by leveraging deep learning to classify 8 distinct weed categories using over 17,000 RGB images. The project also features a user-friendly web application interface developed using Streamlit for real-time weed classification.

## 🔍 Objective
The goal is to develop a model that can help farmers automate the identification of various weed species, reducing manual effort and improving crop yield. We experimented on weed datasets by DeepWeed using three CNN architectures namely ResNet-34, ResNet-50 and Inception-v3, and compared their results.

## 💡 Features
* **Multiclass Classification:** Classifies 8 distinct weed categories.
* **Data Augmentation:** Applied augmentation techniques to address data imbalance.
* **Transfer Learning:** Utilized pretrained ResNet-34 & Inception-v3 models for performance benchmarking. 
* **Custom Model:** Built a CNN (ResNet-50) model from scratch architectures for experimentation.
* **Streamlit Integration:** Real-time predictions through a web-based application.

 ## 📂 Dataset
* **Source:** RGB images of various plant weeds.
* **Format**: The data is stored in `.csv` format. 
* **Size:** Over 17,000 images.
* **Categories:** 8 distinct weed types.
* You can download the dataset:
    * From TesnsorFlow dataset hub: [https://www.tensorflow.org/datasets/catalog/deep_weeds](https://www.tensorflow.org/datasets/catalog/deep_weeds), OR 
    * From GitHub- images & dataset: [https://github.com/AlexOlsen/DeepWeeds/tree/master](https://github.com/AlexOlsen/DeepWeeds/tree/master) ; labels: [https://github.com/AlexOlsen/DeepWeeds/tree/master/labels](https://github.com/AlexOlsen/DeepWeeds/tree/master/labels) 

## 🛠️ Tools & Technologies
* **Framework:** PyTorch
* **Models:** Pre-trained ResNet-34 & Inception-v3, and Custom ResNet-50 
* **Front-End:** Streamlit
* **Programming Language:** Python
* **Additional Libraries:** NumPy, Pandas, Matplotlib, Torchvision
* **Others:** OpenCV, Scikit-Learn

## 📊 Methodology
1. **Data Preprocessing:**
   * Data cleaning and augmentation techniques.
   * Resizing images to a uniform input size.
   * Normalization for consistency across the dataset.

2. **Model Development:**
   * **Custom Model:** Designed a customed ResNet-50 model from scratch & finetuned it for comparison.
     * ResNet-50 'custom model' file: [ResNet-50 from scratch](/1-resnet50-from-scratch-plain-final.ipynb)
     * ResNet-50 'custom model with data augmentation' file: [ResNet-50 from scratch with data augmentation](/2-resnet50-from-scratch-data-augmentation-final.ipynb)
     * ResNet-50 'custom model with pre-trained weights' file: [ResNet-50 from scratch with pre-trained weights](/3-resnet50-from-scratch-pre-trained-weight-final.ipynb)
   * **Transfer Learning:** Utilized pretrained ResNet-34 & Inception-v3 models on ImageNet.
     * Pre-trained ResNet-34 model file: [Pre-trained ResNet-34](/4-resnet34-pre-trained-model-final.ipynb)
     * Pre-trained Inception-v3 model file: [Pre-trained Inception-v3](/5-inception-v3-pre-trained-model-final.ipynb)

3. **Training & Evaluation:**
   * Loss function: CrossEntropyLoss
   * Optimizer: Adam
   * Metrics: Confusion matrix, ROC/AUC curves & F1-Scores
   
4. We've also experimented by applying a **Ray Tune** to find out the best hyperparameters for our model: [Ray Tune](/6_ray_tuner_final.ipynb)
   
5. **Deployment:**
   * Developed a web interface using Streamlit for real-time predictions.

## 📈 Key Results
**CNN Model Performance Analysis:**

* The graphs in the below image present the performance of 3 different Convolutional Neural Network (CNN) architectures, i.e ResNet-50 with 3 variations, and pretrained ResNet-34 & Inception-v3 models--during training and validation on the given dataset. Each line represents a distinct model, and we can observe their training and validation accuracies and losses over 100 epochs.

![Screenshot of the graphs of 'train-val accuracies-losses for all the models'](/train-val-accuracies-losses-for-all-models.png)

* <ins>Overall Observations:</ins>
  * **Pretrained Models:** Using pre-trained weights generally leads to better performance, as evidenced by the higher accuracies and lower losses of the Pretrained-weights ResNet50, Pretrained ResNet34, and Pretrained Inception-v3 models compared to the Plain ResNet50.
  * **Data Augmentation:** Data augmentation can help prevent overfitting, as seen in the improved validation accuracy of the Data Augment ResNet50 compared to the Plain ResNet50.
  * **Architecture Choice:** The choice of architecture also plays a role. ResNet50 and ResNet34 seem to be better suited for this dataset than Inception-v3.

(INSERT A TABLE)

* <ins>Key Findings:</ins>   
  * **ResNet50 (Pretrained + Data Augmentation):** Outperforms all models, achieving the best train/validation accuracy and lowest loss.
  * **Pretrained ResNet34 vs. Inception-v3:** ResNet34 shows better generalization and accuracy than Inception-v3 due to its residual connections, which enable deeper and more effective training.
  * **Plain ResNet50:** Highlights the challenges of training deep models from scratch with imbalanced datasets, including overfitting and lower generalization.

## 🎥 Demo
Click here, [Weed Image Classification Demo](/Streamlit_Agri_Project.gif) , to watch the demo video of the web application.
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


