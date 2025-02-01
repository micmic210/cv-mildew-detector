# Mildew Detection in Cherry Leaves

## Project Overview
This project aims to address the challenges faced by **Farmy & Foods** in detecting powdery mildew on cherry leaves. Powdery mildew is a fungal disease that compromises the quality of cherry crops. Currently, the detection process is manual and time-consuming, taking approximately 30 minutes per tree. This manual method is neither scalable nor efficient, given the large number of cherry trees across multiple farms.

The goal of this project is to develop a **Machine Learning (ML)-powered dashboard** to:

1. **Visually differentiate** between healthy cherry leaves and those affected by powdery mildew.
2. **Predict the health status** of a cherry leaf based on an uploaded image.

By automating this process, the company can significantly reduce time and labor costs, while improving the scalability and accuracy of mildew detection. If successful, this system can be replicated across other crops to detect pests and diseases. 

---

## Dataset Content

The dataset contains images of cherry leaves categorized into two classes:
- **Healthy leaves**
- **Leaves with powdery mildew**

The images were captured from Farmy & Foods’ cherry crops and are publicly available on Kaggle.

- **Dataset Source**: [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Dataset Size**: 4208 images

---

## Business Requirements

The client has outlined two primary business requirements for this project:

1. **Visual Analysis**: Conduct a study to visually differentiate healthy cherry leaves from those with powdery mildew.
2. **Classification and Prediction**: Develop a binary classification model to predict whether a given cherry leaf is healthy or affected by powdery mildew.

---

## Hypotheses and Validation

### Hypotheses
1. **Visual Differences**: 
  - **Healthy leaves** have a uniform texture and color**. 
  - **Mildew-infected leaves** display **discoloration and fungal growth**.
2. **Predictive Feasibility**
  - A **supervised ML model** can classify cherry leaves into healthy or mildew-affected categories **with high accuracy**.

### **Validation**
- **Hypothesis 1**:  
  - Use **mean and variability images** to analyze visual differences.  
  - Generate **image montages** to compare healthy vs. infected leaves.  
- **Hypothesis 2**:  
  - Train a **CNN model** and a **Random Forest classifier**.  
  - Evaluate models using **accuracy, precision, recall, and confusion matrix**.  

---

## The rationale to map the business requirements to the Data Visualisations and ML tasks

| **Business Requirement**    | **Task**  |
|----------------------------|----------------------------------|
| **Visual Analysis**        | - Compute **mean and standard deviation images**.<br>- Compare **healthy vs. infected leaves visually**.<br>- Generate **image montages** to highlight differences. |
| **Classification & Prediction** | - Train a **CNN model** with optimized hyperparameters.<br>- **Validate model performance** on test data.<br>- Deploy a **Streamlit dashboard** for real-time predictions. |

### **Business Requirement 1**: Visual Analysis
- Display the mean and standard deviation images for healthy and mildew-affected leaves.
- Showcase differences between the average healthy and mildew-affected leaves.
- Create an image montage to illustrate the dataset's diversity.

### **Business Requirement 2**: Classification and Prediction
- Develop a binary classifier to predict whether a leaf is healthy or has mildew.
- Build a user-friendly interface for uploading images and receiving predictions in real time.

---

## ML Business Case

### **Mildew Detection Model**
- **Objective**: Predict if a cherry leaf is healthy or has powdery mildew.
- **Model Type**: Supervised binary classification.
- **Success Metrics**:
  - Accuracy ≥ 90%
  - Confusion Matrix: High recall for mildew-affected leaves.
- **Model Output**: Probability of the leaf being healthy or mildew-affected, along with a classification label.

**Heuristics**:
Currently, the manual inspection process relies on human expertise, which is prone to errors and inefficiencies. An ML-based solution can provide faster, more consistent results, minimizing human error and operational costs.

**Training Data**:
The training data is derived from the cherry leaves dataset, consisting of labeled images for healthy and mildew-affected categories.

---

## **Project Workflow: CRISP-DM Framework**

1. **Business Understanding**  
   - **Problem**: **Manual detection is inefficient and unscalable**.  
   - **Goal**: Develop an **ML-based mildew detection system**.  
2. **Data Understanding & Preparation**  
   - Dataset: **4,208 labeled images**.  
   - Processing: **Image cleaning, resizing, augmentation**.  
3. **Modeling**  
   - **Train CNN & Random Forest models**.  
   - **Optimize hyperparameters** for best performance.  
4. **Evaluation**  
   - **Compare model performance** using precision, recall, and accuracy.  
5. **Deployment**  
   - **Deploy an interactive AI-powered Streamlit dashboard**.  

---

## **Dashboard Design**
The **Streamlit dashboard** provides an **interactive AI-powered tool** for visualizing cherry leaf data and predicting mildew infections. It is structured into five key pages.

---

### **Page 1: Quick Project Summary**
- **Business Requirement**: Provide an overview of the project and its objectives.
- **Content**:
  - **Introduction to Powdery Mildew**:  
    - Explanation of powdery mildew as a fungal disease.
    - Its impact on cherry crops, including reduced yield and quality.
  - **Current Detection Process & Its Limitations**:  
    - Manual inspection takes **~30 minutes per tree**, which is slow and unscalable.
    - Prone to **human error and inconsistency**.
  - **Project Goals**:  
    - Develop a **machine learning model** to automate mildew detection.
    - Provide an **interactive dashboard** for real-time classification.
  - **How the Solution Works**:  
    - Users can **upload images of cherry leaves** for classification.
    - Model predicts **Healthy** or **Infected** status.
    - Dashboard provides **data insights and visual comparisons**.

---

### **Page 2: Leaf Visualizer**
- **Business Requirement**: Address the visual differentiation of healthy vs. mildew-affected leaves.
- **Features**:
  - **Checkbox 1**: Display **mean and standard deviation images** for both categories.  
    - Users can observe the **average appearance** of healthy and infected leaves.  
  - **Checkbox 2**: Show **differences between the average images** of healthy vs. infected leaves.  
    - A **pixel-wise difference image** highlights **distinctive mildew patterns**.  
  - **Checkbox 3**: Display an **image montage** of the dataset.  
    - Users can view **random samples** of both **healthy** and **infected** leaves.  
    - Helps confirm **dataset quality and diversity**.

---

### **Page 3: Mildew Detector**
- **Business Requirement**: Predict the health status of a cherry leaf based on an uploaded image.
- **Features**:
  - **Upload Widget**:  
    - Users can **upload single or multiple images** of cherry leaves.
    - Accepted formats: `.jpg`, `.png`, `.jpeg`.
  - **Real-Time Predictions**:  
    - Model classifies each image as **Healthy** or **Infected**.
    - Displays **confidence score (%)** for each prediction.
  - **Prediction Results Table**:  
    - Shows **file name, predicted class, and confidence score**.
    - Allows **sorting and filtering** results.
  - **Download Button**:  
    - Users can download the **prediction results as a CSV file** for future reference.

---

### **Page 4: Project Hypotheses and Validation**
- **Business Requirement**: Demonstrate how hypotheses were tested and validated.
- **Content**:
  - **Hypothesis 1: Visual Differences Exist**  
    - Display **mean and standard deviation images**.
    - Provide **side-by-side comparisons** of healthy vs. infected leaves.
  - **Hypothesis 2: AI Can Predict Mildew Infections**  
    - Show **model performance metrics** (e.g., accuracy, precision, recall).
    - Provide **example predictions** with confidence scores.
  - **Supporting Visualizations**:  
    - Side-by-side comparisons of **healthy vs. infected** leaves.
    - Sample **classification results** from the model.
    - Explanation of **data preprocessing and augmentation techniques**.

---

### **Page 5: ML Performance Metrics**
- **Business Requirement**: Provide insights into the model’s accuracy and effectiveness.
- **Content**:
  - **Dataset Summary**:  
    - **Label frequencies** in train, validation, and test sets.
    - **Bar charts** showing class distribution across datasets.
  - **Training History**:  
    - Line graphs of **training vs. validation accuracy** over epochs.
    - **Loss curves** to track model convergence.
  - **Evaluation Metrics**:  
    - **Confusion Matrix** visualization.  
    - **Classification Report** (Precision, Recall, F1-score).  
    - **Model comparison table**: CNN vs. Random Forest.  
  - **Model Performance Takeaways**:  
    - Key observations on **misclassified samples**.
    - Strengths and limitations of the chosen model.  

---


## **User Stories**
The **Streamlit dashboard** should provide an interactive and user-friendly experience for visualizing cherry leaf conditions, making predictions, and understanding model performance.

### 1. **Intuitive Navigation**
   - **Priority**: Must-Have  
   - **User Story**:  
     As a user, I want an intuitive dashboard with clear navigation so that I can easily access all features and insights.  
   - **Acceptance Criteria**:  
     - A **navigation bar** is present to allow switching between all pages.  
     - All navigation links are **clearly labeled and functional**.  
     - The user can reach any page in **two or fewer clicks**.  
     - The UI layout is **consistent and responsive** across devices.  

### 2. **Visual Differentiation**  
   - **Priority**: Must-Have  
   - **User Story**:  
     As a user, I want to visually compare healthy and mildew-affected leaves so that I can understand the key differences.  
   - **Acceptance Criteria**:  
     - **Checkbox 1**: Display **mean and variability images** for both classes.  
     - **Checkbox 2**: Show **side-by-side comparisons** of average healthy vs. mildew-affected leaves.  
     - **Checkbox 3**: Display an **image montage** showing multiple examples from each class.  
     - All visualizations are **clearly labeled** and easy to interpret.  

### 3. **Image Montage**  
   - **Priority**: Nice-to-Have  
   - **User Story**:  
     As a user, I want to view multiple examples of healthy and infected leaves in a montage so that I can analyze visual patterns.  
   - **Acceptance Criteria**:  
     - A **dropdown menu** allows users to select "Healthy" or "Infected" leaves.  
     - The montage displays at least **9 images per category** in a **grid format**.  
     - Users can **regenerate** the montage with new random images.  

### 4. **Real-Time Predictions**  
   - **Priority**: Must-Have  
   - **User Story**:  
     As a user, I want to upload images of cherry leaves and receive predictions about their health status so that I can detect mildew quickly.  
   - **Acceptance Criteria**:  
     - A **file uploader** allows users to select **single or multiple images**.  
     - The model predicts whether a leaf is **Healthy** or **Infected** with at least **90% accuracy**.  
     - Results include:  
       - **Classification label** (Healthy/Infected).  
       - **Confidence score** for each prediction.  
     - Users can download a **summary report** with image names and predictions.  

### 5. **Infection Rate Summary**  
   - **Priority**: Nice-to-Have  
   - **User Story**:  
     As a user, I want to see a summary of the infection rate (percentage of healthy vs. infected leaves) based on my uploaded images so that I can quickly assess the situation.  
   - **Acceptance Criteria**:  
     - A **dynamic pie chart** displays the proportion of healthy vs. infected leaves.  
     - The chart **updates automatically** when new images are uploaded.  
     - Infection rates are **displayed as percentages**.  
     - Users can download a **summary table** of infection rates.  

### 6. **Hypotheses and Validation**  
   - **Priority**: Must-Have  
   - **User Story**:  
     As a stakeholder, I want to understand the hypotheses tested and their validation process so that I can trust the project results.  
   - **Acceptance Criteria**:  
     - A dedicated **Hypotheses & Validation** page explains the tested hypotheses.  
     - Visualizations support **hypothesis validation** with comparisons of healthy vs. infected leaves.  
     - Key insights are clearly summarized.  

### 7. **Model Performance Metrics**  
   - **Priority**: Must-Have  
   - **User Story**:  
     As a stakeholder, I want to see detailed performance metrics so that I can evaluate how reliable the model is.  
   - **Acceptance Criteria**:  
     - The dashboard displays:  
       - **Confusion matrix** with actual vs. predicted labels.  
       - **Classification report** (precision, recall, F1-score).  
       - **Training history** (accuracy and loss curves).  
     - All metrics are **clearly explained** and easy to interpret.  

### 8. **Scalability and Future Applications**  
   - **Priority**: Nice-to-Have  
   - **User Story**:  
     As a stakeholder, I want to explore how this project could be expanded to detect other plant diseases so that I can assess its scalability.  
   - **Acceptance Criteria**:  
     - A **"Future Applications"** section outlines possible expansions (e.g., detecting diseases in other crops).  
     - The section discusses **model retraining** with new datasets.  
     - Potential **IoT and drone-based integrations** are briefly outlined.  

---

## Unfixed Bugs

- The model’s performance may vary under non-standard conditions (e.g., unusual lighting or damaged leaves)

---

## Deployment

### Heroku


- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- The project was deployed to Heroku following these simplified steps:

1. Log in to Heroku and create an app.
2. Link the app to the GitHub repository containing the project code.
3. Select the branch to deploy and click "Deploy Branch."
4. Once the deployment completes, click "Open App" to access the live app.
5. Ensure that deployment files, such as `Procfile` and `requirements.txt`, are correctly configured.
6. Use a `.slugignore` file to exclude unnecessary large files if the slug size exceeds limits.

### Repository Structure
- **app_pages/**: Streamlit app pages.
- **src/**: Auxiliary scripts (e.g., data preprocessing, model evaluation).
- **notebooks/**: Jupyter notebooks for data analysis and model training.
- **Procfile, requirements.txt, runtime.txt, setup.sh**: Files for Heroku deployment.

---

## Technologies Used
- **Python Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, TensorFlow/Keras, Streamlit.
- **Tools**: Jupyter Notebook, Heroku for deployment.

---

## Future Work
- Extend the system to detect other crop diseases.
- Incorporate real-time image capture from drones for automated data collection.
- Integrate the system with IoT devices for automated spraying of antifungal compounds.

---

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

---

## Credits

- The deployment steps were adapted from [Heroku Documentation](https://devcenter.heroku.com/).
- Data preprocessing techniques were inspired by [TensorFlow tutorials](https://www.tensorflow.org/tutorials).
- Model evaluation approaches referenced [Scikit-Learn Documentation](https://scikit-learn.org/stable/).
- Icons used in the dashboard are from [Font Awesome](https://fontawesome.com/).
- Visualization techniques were guided by examples from [Matplotlib Documentation](https://matplotlib.org/stable/index.html).

---

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

---

### Media

---


## Acknowledgements
- **Farmy & Foods** for providing the dataset and project inspiration.
- Code Institute for guidance and support in building this project.
- Kaggle for hosting the cherry leaves dataset and enabling access to quality data.
- The contributors of TensorFlow and Scikit-Learn for their excellent documentation and tutorials.
- Community forums and online resources for addressing technical challenges and sharing best practices.

---
