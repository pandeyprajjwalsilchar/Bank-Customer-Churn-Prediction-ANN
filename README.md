Customer Churn Prediction using Artificial Neural Network (ANN)
This project involves building and training an Artificial Neural Network (ANN) to predict customer churn using structured customer data. The model identifies customers who are likely to stop using a service, enabling businesses to take proactive retention measures.

Project Features
End-to-End Pipeline: Includes data preprocessing, model training, evaluation, and visualization.
Binary Classification: Predicts whether a customer will churn (1) or not (0).
Neural Network Implementation: Built using TensorFlow/Keras with ReLU activation for hidden layers and sigmoid activation for the output layer.
Real-World Dataset: The Churn_Modelling.csv dataset contains 10,000 customer records with features like geography, gender, age, and balance.
Project Workflow
Import Libraries

Libraries used: numpy, pandas, tensorflow, sklearn, and matplotlib.

Data Preprocessing:
  Load the dataset.
  Handle categorical features:
    Use LabelEncoder for gender.
    Apply OneHotEncoder for geography.
  Split the data into training and test sets.
  Standardize the features using StandardScaler for uniform scaling.
  
Building the ANN
  Initialize the model with Sequential.
  Add:
    Two hidden layers with ReLU activation.
    One output layer with a sigmoid activation function.
  Compile the model using the Adam optimizer and binary cross-entropy loss.

Training the Model
  Train the ANN on the training data for 100 epochs with a batch size of 32.
  Monitor training loss and accuracy.

Model Evaluation
  Test the ANN on unseen data.
  Evaluate performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

Project Highlights:
  Accuracy: Achieved 86% accuracy on the test dataset.
  Impact: Predicts customer churn to help businesses reduce attrition rates.
  Technology Stack: Python, TensorFlow, Keras, Pandas, Scikit-learn.

Results:
  Training Accuracy: 88%
  Test Accuracy: 86%
  Confusion Matrix:
  True Positive Rate: 0.85
  False Positive Rate: 0.15
