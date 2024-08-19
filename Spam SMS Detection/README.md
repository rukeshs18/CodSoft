# Spam SMS Detection

This project focuses on building a spam SMS detection system using a Naive Bayes classifier. The process includes data preprocessing, feature extraction, model training, evaluation, and visualization. Below is a detailed description of each step:

## 1. **Importing Libraries**

The following libraries are imported for the project:

- **pandas**: For data manipulation and analysis.
- **matplotlib.pyplot** and **seaborn**: For data visualization.
- **sklearn.model_selection**: For splitting the dataset into training and testing sets.
- **sklearn.feature_extraction.text**: For converting text data into numerical features using TF-IDF.
- **sklearn.naive_bayes**: For applying the Naive Bayes classification algorithm.
- **sklearn.metrics**: For evaluating the performance of the model.

## 2. **Loading the Dataset**

The dataset is loaded from a CSV file using `pandas`. The dataset contains two columns: the label of the message (`v1`) and the message content itself (`v2`). The relevant columns are renamed to 'label' and 'message' for clarity.

## 3. **Preprocessing**

- **Label Encoding**: The 'label' column is mapped from categorical values ('ham' and 'spam') to binary values (0 for 'ham' and 1 for 'spam'). This is necessary for the classification algorithm to process the labels.

## 4. **Splitting the Dataset**

The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`. The split ratio is set to 75% for training and 25% for testing, with a random seed for reproducibility.

## 5. **Feature Extraction Using TF-IDF**

- **TF-IDF Vectorization**: The `TfidfVectorizer` is used to convert the text messages into numerical features. The vectorizer transforms the text data into TF-IDF (Term Frequency-Inverse Document Frequency) features, which reflect the importance of words in the documents.
- **Stop Words and Maximum Document Frequency**: Common English stop words are ignored, and terms that appear in more than 70% of the documents are excluded to reduce feature dimensionality.

## 6. **Training the Naive Bayes Classifier**

- **Model Training**: A `MultinomialNB` classifier is trained on the TF-IDF features of the training data. Naive Bayes is well-suited for text classification tasks like spam detection.

## 7. **Making Predictions**

- **Model Prediction**: The trained model is used to predict the labels of the test set messages. The predictions are compared to the true labels to evaluate the model's performance.

## 8. **Evaluating the Model**

- **Accuracy**: The accuracy of the model is calculated to determine the proportion of correctly classified messages.
- **Confusion Matrix**: A confusion matrix is generated to show the counts of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: A detailed classification report is produced, including precision, recall, and F1-score for both classes ('ham' and 'spam').

## 9. **Visualization**

1. **Distribution of Messages by Label**: A bar plot is created to visualize the distribution of messages in the dataset, showing the number of 'ham' and 'spam' messages.

2. **Confusion Matrix**: A heatmap of the confusion matrix is plotted to visualize the model's performance in classifying messages as 'ham' or 'spam'.

3. **Top Words in Spam and Ham**:
   - **Feature Log Probabilities**: The top words contributing to the classification of messages as 'ham' or 'spam' are identified based on their log probabilities.
   - **Top Words Plot**: Horizontal bar plots are used to visualize the top 20 words in 'ham' and 'spam' messages, highlighting the most significant terms for each class.

4. **Histogram of Message Lengths**: A histogram is plotted to show the distribution of message lengths for 'ham' and 'spam' messages. This provides insight into whether message length is a distinguishing feature between the two classes.

## 10. **Testing with a Custom Message**

- **Sample Prediction**: A custom message is tested with the trained model to predict whether it is 'spam' or 'ham'. The result demonstrates the model’s capability to classify new, unseen messages.

## **Conclusion**

The spam SMS detection project illustrates a typical pipeline for text classification:

1. **Data Preparation**: Includes loading, preprocessing, and transforming text data into numerical features suitable for machine learning.
2. **Model Training and Evaluation**: Uses a Naive Bayes classifier to build and assess a spam detection model, showcasing key metrics like accuracy and confusion matrix.
3. **Visualization**: Provides insights into data distribution, model performance, and important features, helping to understand the effectiveness of the spam detection system.

Overall, the project demonstrates how to effectively apply machine learning techniques to text classification tasks, particularly for detecting spam messages. Future improvements could involve experimenting with other classifiers, incorporating additional features, or optimizing hyperparameters for enhanced performance.
