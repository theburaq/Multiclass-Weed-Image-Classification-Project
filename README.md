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
1. **Exploratory Data Analysis:**
   * All 17,509 labelled images from DeepWeeds were partitioned into 60%-20%-20% splits of training, validation and testing subsets for k-fold cross validation with k = 5. More details about the process is mentioned here in the research paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/#Sec2](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/#Sec2) .
   * The selected weed species are local to pastoral grasslands across the state of Queensland. They include: "Chinee apple", "Snakeweed", "Lantana", "Prickly acacia", "Siam weed", "Parthenium", "Rubber vine" and "Parkinsonia".
     ![Images of the sample weeds](/sample-weeds-image-1.png)


   * Over 1,000 images were collected of each weed species, totaling over 8,000 images of positive species classes. Images of neighbouring flora and backgrounds that did not contain the weed species of interest were collated into a single “negative” class.
   * From bar charts and data, it's clear there's a class imbalance issue, with the "Negative" class having significantly more samples (9106) compared to other species (around 1000-1100 each):
     ![Screenshot of the graphs of 'Class Imbalance'](/class-imbalance-graph.png)

  
2. **Data Preprocessing:**
   * Data cleaning and augmentation techniques.
   * Resizing images to a uniform input size.
   * Normalization for consistency across the dataset.

3. **Model Development:**
   * **Custom Model:** Designed a customed ResNet-50 model from scratch & finetuned it for comparison.
     * ResNet-50 'custom model' file: [ResNet-50 from scratch](/1-resnet50-from-scratch-plain-final.ipynb)
     * ResNet-50 'custom model with data augmentation' file: [ResNet-50 from scratch with data augmentation](/2-resnet50-from-scratch-data-augmentation-final.ipynb)
     * ResNet-50 'custom model with pre-trained weights' file: [ResNet-50 from scratch with pre-trained weights](/3-resnet50-from-scratch-pre-trained-weight-final.ipynb)
   * **Transfer Learning:** Utilized pretrained ResNet-34 & Inception-v3 models on ImageNet.
     * Pre-trained ResNet-34 model file: [Pre-trained ResNet-34](/4-resnet34-pre-trained-model-final.ipynb)
     * Pre-trained Inception-v3 model file: [Pre-trained Inception-v3](/5-inception-v3-pre-trained-model-final.ipynb)

4. **Training & Evaluation:**
   * Loss function: CrossEntropyLoss
   * Optimizer: Adam
   * Metrics: Confusion matrix, ROC/AUC curves & F1-Scores
   
5. I've also experimented by applying a **Ray Tune** to find out the best hyperparameters for our model: [Ray Tune](/6_ray_tuner_final.ipynb)
   
6. **Deployment:**
   * Developed a web interface using Streamlit for real-time predictions.

## 📈 Key Results
**CNN Model Performance Analysis:**

* The graphs in the below image present the performance of the different Convolutional Neural Network (CNN) architectures, i.e ResNet-50 with 3 variations, and pretrained ResNet-34 & Inception-v3 models--during training and validation on the given dataset. Each line represents a distinct model, and we can observe their training and validation accuracies and losses over 100 epochs.

![Screenshot of the graphs of 'train-val accuracies-losses for all the models'](/train-val-accuracies-losses-for-all-models.png)

* <ins>Comparison and Key Insights:</ins>
1. **Plain ResNet-50:**
   * Exhibited overfitting: Training accuracy soared, but validation accuracy lagged significantly (~60%).
   * The validation loss was consistently high compared to other models.

2. **Data-Augmented ResNet-50:**
   * Data augmentation greatly enhanced validation accuracy to 85% and reduced overfitting.
   * Losses showed much smoother convergence compared to the plain model.

3. **Pretrained ResNet-50 (Weights):**
   * Fine-tuning a pretrained ResNet-50 model was highly effective, achieving 90% validation accuracy.
   * Loss curves were smooth and demonstrated faster convergence.

4. **Pretrained ResNet-34:**
   * Performed well but slightly trailed the pretrained ResNet-50 in both accuracy and loss metrics.

5. **Pretrained Inception-v3:**
   * Top-performing model, achieving the highest validation accuracy (92%) and the lowest loss values.
   * Particularly excelled in identifying minority classes, likely due to its multiscale feature extraction capabilities.

## 🎥 Demo
Click here, [Weed Image Classification Demo](/Streamlit_Agri_Project.gif) , to watch the demo video of the web application.
* Features in the demo:
  * Upload an image of a weed.
  * Real-time classification display.
  * Simple and intuitive user interface.
 
 ## 🚀 Future Improvements
* Extend the dataset to include additional weed species.
* Exploring techniques like class weighting, oversampling, or SMOTE could further enhance model performance.
* Modern architectures like EfficientNet or Vision Transformers (ViTs) could be tested for this task.
* Experiment with ensemble methods combining ResNet-50 and Inception-v3 for improved performance. Incorporate attention mechanisms to focus on class-specific features.
* Visualizing model decisions using Grad-CAM or similar tools could provide insights into the features learned by the CNNs.
* Deploy the Streamlit app on a cloud platform for wider accessibility.

## 🪴 Conclusion
This project demonstrated the potential of CNN architectures, particularly pretrained models like Inception-v3, in tackling multiclass weed classification. While significant strides were made in achieving high accuracy and generalization, challenges like class imbalance remain. Future work could focus on augmenting minority classes and leveraging advanced architectures for even better results.

## 📘 References
* The Research paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6375952/)
* [https://paperswithcode.com/paper/deepweeds-a-multiclass-weed-species-image/review/](https://paperswithcode.com/paper/deepweeds-a-multiclass-weed-species-image/review/)

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


