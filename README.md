# CodSoft Internship

This project demonstrates a series of data analysis and machine learning tasks applied to various datasets. The goal is to showcase different techniques for classification, preprocessing, model evaluation, and visualization. Below, we describe the key components of the project and the code used to implement them.

## 1. Spam SMS Detection with Naive Bayes

**Objective:** Build a text classification model to distinguish between spam and ham (non-spam) SMS messages.

**Key Steps:**
1. **Data Loading:** Load the SMS dataset and select relevant columns.
2. **Preprocessing:** Convert categorical labels to binary values (0 for ham and 1 for spam).
3. **Data Splitting:** Split the data into training and test sets.
4. **Feature Extraction:** Use TF-IDF to convert text messages into numerical features.
5. **Model Training:** Train a Multinomial Naive Bayes classifier on the training data.
6. **Prediction & Evaluation:** Evaluate the model’s performance using accuracy, confusion matrix, and classification report.
7. **Visualization:** 
   - Plot the distribution of messages by label.
   - Visualize the confusion matrix.
   - Display top words in ham and spam messages.
   - Show a histogram of message lengths.

## 2. Customer Churn Prediction with Logistic Regression

**Objective:** Predict customer churn using logistic regression and evaluate model performance.

**Key Steps:**
1. **Data Loading:** Load and preprocess the customer churn dataset.
2. **Data Preprocessing:** Encode categorical variables and scale features.
3. **Data Splitting:** Divide the data into training and test sets.
4. **Model Training:** Train a Logistic Regression model on the training data.
5. **Model Evaluation:** Assess the model using accuracy, confusion matrix, classification report, and ROC-AUC score.
6. **Hyperparameter Tuning (Optional):** Perform grid search to find the best hyperparameters.
7. **Visualization:** 
   - Plot the confusion matrix.
   - Display the ROC curve.
   - Show feature importance based on model coefficients.
   - Visualize learning curves.
   - Plot the precision-recall curve.
   - Display a histogram of predicted probabilities.

## 3. Credit Card Fraud Detection with Logistic Regression and SMOTE

**Objective:** Detect fraudulent credit card transactions using logistic regression and address class imbalance with SMOTE.

**Key Steps:**
1. **Data Loading:** Load and preprocess the credit card fraud dataset.
2. **Data Preprocessing:** Scale features, encode categorical variables, and drop irrelevant columns.
3. **Class Imbalance Handling:** Use SMOTE to balance the dataset.
4. **Model Training:** Train a Logistic Regression model within a pipeline that includes scaling.
5. **Model Evaluation:** Evaluate the model using confusion matrix, classification report, accuracy, and ROC-AUC score.
6. **Visualization:**
   - Plot the confusion matrix.
   - Display the ROC curve.

## Conclusion

This internship project provided hands-on experience in applying machine learning techniques to real-world problems. Through the analysis of SMS messages, customer churn, and credit card transactions, we demonstrated essential skills in data preprocessing, model training, and evaluation.

- **Spam SMS Detection:** We utilized TF-IDF and Naive Bayes to effectively classify SMS messages, revealing the importance of feature extraction and model evaluation in text classification tasks.
- **Customer Churn Prediction:** By leveraging logistic regression and hyperparameter tuning, we highlighted the impact of model optimization and feature importance in predicting customer behavior.
- **Credit Card Fraud Detection:** Through the application of SMOTE and logistic regression, we addressed class imbalance issues, showcasing the significance of balancing datasets for fraud detection.

Overall, the project underscores the value of thorough data analysis, feature engineering, and model evaluation in building robust and reliable machine learning solutions. The use of various visualization techniques further aids in understanding model performance and insights gained from the data.
