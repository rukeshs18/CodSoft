# Customer Churn Prediction

This project involves predicting whether customers will churn using a Logistic Regression model. The process encompasses data preprocessing, model training, evaluation, hyperparameter tuning, and visualization. Here’s a detailed overview of each step:

## 1. **Loading the Dataset**

The dataset is loaded from a CSV file containing customer information and their churn status. This dataset provides the foundation for the analysis and model development.

## 2. **Data Preprocessing**

- **Label Encoding**: Categorical variables such as 'Gender' and 'Geography' are transformed into numerical values. This step is essential for converting non-numeric data into a format suitable for machine learning models.

- **Feature and Target Definition**: The dataset is prepared by defining features (independent variables) and the target variable (dependent variable). Unnecessary columns, such as customer identifiers and names, are removed to focus on relevant data.

## 3. **Feature Scaling**

Feature scaling is applied to standardize the data, ensuring that all features contribute equally to the model. This is achieved using a standard scaler that normalizes the features to have a mean of zero and a standard deviation of one.

## 4. **Splitting the Dataset**

The data is divided into training and testing sets. This split allows the model to be trained on one subset of the data while being evaluated on another to assess its performance.

## 5. **Training the Logistic Regression Model**

A Logistic Regression model is trained using the training data. This model is used to predict the likelihood of customer churn based on the provided features.

## 6. **Model Evaluation**

The performance of the Logistic Regression model is assessed using various metrics:
- **Accuracy**: Measures the proportion of correctly predicted instances.
- **Confusion Matrix**: Displays the counts of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **ROC-AUC Score**: Evaluates the model’s ability to distinguish between the churn and non-churn classes.

## 7. **Hyperparameter Tuning (Optional)**

Hyperparameter tuning involves using techniques like Grid Search to identify the best parameters for the Logistic Regression model. This process helps optimize the model’s performance by testing different values for regularization strength, penalty type, and solver type.

## 8. **Retrain Model with Best Parameters (Optional)**

Once the best hyperparameters are identified, the model is retrained using these optimal settings. The performance of this tuned model is then evaluated to compare improvements over the initial model.

## 9. **Visualization**

Visualizations are created to better understand the model’s performance and feature importance:
- **Confusion Matrix**: A heatmap visualizes the confusion matrix to show the model’s performance across different classes.
- **ROC Curve**: Plots the Receiver Operating Characteristic (ROC) curve to illustrate the trade-off between the true positive rate and false positive rate.
- **Feature Importance Plot**: Displays the importance of each feature based on the magnitude of the model’s coefficients.
- **Learning Curves**: Shows how the model’s performance improves with more training data.
- **Precision-Recall Curve**: Evaluates the trade-off between precision and recall.
- **Histogram of Predicted Probabilities**: Illustrates the distribution of predicted probabilities for the positive class.

## **Conclusion**

The customer churn prediction project provides a thorough approach to identifying customers likely to churn using a Logistic Regression model. The process includes essential steps such as data preprocessing, model training, and evaluation, supplemented by hyperparameter tuning and insightful visualizations. These steps collectively aid in understanding and improving the model’s performance, with potential for further refinement through advanced techniques or additional data.
