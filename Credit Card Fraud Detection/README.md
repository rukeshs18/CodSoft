# Credit Card Fraud Detection

This project demonstrates a comprehensive approach to detecting fraudulent credit card transactions using a Logistic Regression model. It encompasses data preprocessing, handling class imbalance, model training, and evaluation. Below is a detailed explanation of each step:

## 1. **Importing Libraries**

Various libraries are imported for data manipulation, visualization, machine learning, and handling imbalanced datasets:

- **pandas** for data manipulation.
- **numpy** for numerical operations.
- **matplotlib.pyplot** and **seaborn** for data visualization.
- **sklearn** for machine learning algorithms and metrics.
- **imblearn** for handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique).
- **sklearn.pipeline** for creating streamlined pipelines.

## 2. **Loading the Dataset**

The dataset, which contains information about credit card transactions, is loaded from a CSV file into a DataFrame.

## 3. **Displaying Column Names and Checking for Missing Values**

Column names are displayed to understand the dataset's structure, and checks are performed for missing values to ensure the data is complete.

## 4. **Feature Scaling**

The transaction amount feature is scaled using standardization, which transforms the feature to have a mean of 0 and a standard deviation of 1. This helps in improving the model's performance.

## 5. **Dropping Irrelevant Columns**

Columns that are not useful for the fraud detection task are removed to focus on relevant features.

## 6. **Encoding Categorical Variables**

Categorical variables with high cardinality are encoded using label encoding. Low-cardinality categorical variables are one-hot encoded. This converts categorical data into numerical format suitable for machine learning algorithms.

## 7. **Separating Features and Target Variable**

The dataset is divided into feature variables (X) and the target variable (y), where the target variable indicates whether a transaction is fraudulent or legitimate.

## 8. **Visualizing Data Distribution**

Visualizations are created to understand the distribution of transaction amounts and the class distribution (legitimate vs. fraudulent) before applying data balancing techniques. This helps in understanding the dataset's characteristics and identifying any imbalances.

## 9. **Splitting Data into Train and Test Sets**

The data is split into training and test sets, with a specified percentage reserved for testing. This allows for the evaluation of the model’s performance on unseen data.

## 10. **Handling Class Imbalance with SMOTE**

SMOTE is applied to address class imbalance by generating synthetic samples for the minority class (fraudulent transactions). This technique helps in balancing the class distribution in the training set, improving the model's ability to learn from the minority class.

## 11. **Visualizing Class Distribution After SMOTE**

A visualization of the class distribution after applying SMOTE is created to verify that the class imbalance has been addressed.

## 12. **Defining and Training the Logistic Regression Model**

A logistic regression model is defined and trained using the resampled training data. A pipeline is created to streamline the preprocessing (scaling) and modeling steps, ensuring a smooth workflow.

## 13. **Evaluating the Model**

The model’s performance is assessed using several metrics:
- **Confusion Matrix:** Shows counts of true positives, false positives, true negatives, and false negatives, summarizing prediction results.
- **Classification Report:** Provides precision, recall, and F1-score for each class, giving a detailed evaluation of model performance.
- **Accuracy Score:** Measures the proportion of correct predictions.
- **ROC AUC Score:** Reflects the model's ability to distinguish between classes.

## 14. **Visualizing Evaluation Metrics**

**Confusion Matrix:** A heatmap visualizes the confusion matrix, illustrating the performance of the model in terms of correctly and incorrectly classified transactions.

**ROC Curve:** The ROC curve is plotted to show the trade-off between the true positive rate and false positive rate, with the ROC AUC score indicating the model’s overall performance in class discrimination.

## **Conclusion**

The Credit Card Fraud Detection project effectively demonstrates the process of building and evaluating a fraud detection system using logistic regression. Key insights from the project include:

1. **Data Preparation:** Comprehensive data preprocessing, including scaling, encoding, and handling missing values, is essential for effective model training and evaluation.

2. **Class Imbalance:** Addressing class imbalance with techniques like SMOTE improves the model’s ability to detect fraudulent transactions, which is crucial given the typically imbalanced nature of fraud detection datasets.

3. **Model Performance:** The logistic regression model, trained with balanced data, shows strong performance with high accuracy and a robust ROC AUC score, indicating its effectiveness in distinguishing between fraudulent and legitimate transactions.

4. **Visualization:** Visualizations, such as the confusion matrix and ROC curve, offer valuable insights into the model’s performance and its ability to handle fraud detection tasks.

Overall, the project highlights the importance of thorough data preprocessing, effective handling of class imbalance, and rigorous evaluation in developing a reliable fraud detection system. Future work could involve exploring different algorithms, further feature engineering, and incorporating additional data sources to enhance model performance and robustness.
