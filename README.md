# 🌱 Multiclass Weed Image Classification Using Deep Learning 🌾
This is an AgriAI HIVE project done during my Internship with LiveAI.

## 🌟 Project Overview
Weed infestation is a critical challenge in sustainable agriculture, impacting crop yields and increasing reliance on herbicides. This project addresses the need for precise weed identification by leveraging deep learning to classify 8 distinct weed categories using over 17,000 RGB images. The project also features a user-friendly web application interface developed using Streamlit for real-time weed classification.

## 🔍 Objective
The goal is to develop a model that can help farmers automate the identification of various weed species, reducing manual effort and improving crop yield. We experimented on weed datasets by DeepWeed using two CNN architectures namely ResNet-50 and Inception-v3, and compare the results.

## 💡 Features
* **Multiclass Classification:** Classifies 8 distinct weed categories.
* **Data Augmentation:** Applied augmentation techniques to address data imbalance.
* **Transfer Learning:** Utilized pretrained ResNet-50 and Inception-v3 architectures for experimentation.
* **Custom Model:** Built a CNN (ResNet-50) model from scratch for performance benchmarking.
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
* **Models:** ResNet-50, Inception-v3, and a custom CNN
* **Front-End:** Streamlit
* **Programming Language:** Python
* **Additional Libraries:** NumPy, Pandas, Matplotlib, Torchvision

## 📊 Methodology
1. **Data Preprocessing:**
  * Data cleaning and augmentation techniques.
  * Splitting data into training, validation, and testing sets.
  * Resizing images to a uniform input size.
  * Normalization for consistency across the dataset.
2. **Model Development:**
 * **Transfer Learning:** Fine-tuned ResNet-50 and Inception-v3 pretrained on ImageNet.
    * 1_ResNet50_from_scratch_Final.ipynb
    * (ADD THE JUPYTER NB LINKS HERE AFTER UPLOADING)
 * **Custom Model:** Designed a CNN(ResNet-50) from scratch for comparison.
    * (ADD THE JUPYTER NB LINKS HERE AFTER UPLOADING)
3. **Training & Evaluation:**
 * Loss function: CrossEntropyLoss
 * Optimizer: Adam
 * Metrics: Confusion matrix & ROC/AUC curves
 * 
4. We've also experimented by applying a **Ray Tune** to find out the best hyperparameters: (ADD THE JUPYTER NB LINKS HERE AFTER UPLOADING)
   
5. **Deployment:**
 * Developed a web interface using Streamlit for real-time predictions.

#**(EDIT THE BELOW AFTER ULOADING ALL THE FILES)**

📈 Key Results
Best Performing Model: (Mention your top-performing model, e.g., ResNet-50 with transfer learning)
Accuracy: Specify the percentage, e.g., 92.4%
Other Metrics: Precision: X%, Recall: Y%, F1-Score: Z%.

🎥 Demo
Click here to watch the demo video of the web application.
* Features in the demo:
 * Upload an image of a weed.
 * Real-time classification and confidence score display.
 * Simple and intuitive user interface.

📘 References
Weed Detection in Upland Cotton Production Using Deep Learning

🚀 Future Improvements
Extend the dataset to include additional weed species.
Explore advanced architectures like Vision Transformers.
Deploy the Streamlit app on a cloud platform for wider accessibility.
Incorporate explainability tools like Grad-CAM for better interpretability.

🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.



