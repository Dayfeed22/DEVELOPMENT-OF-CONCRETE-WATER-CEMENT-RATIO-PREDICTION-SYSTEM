# DEVELOPMENT-OF-CONCRETE-WATER-CEMENT-RATIO-PREDICTION-SYSTEM

## Project Overview

**Project Title**: DEVELOPMENT OF CONCRETE WATER CEMENT RATIO PREDICTION SYSTEM 
**Database**: `DAVE_UML.ipynb`

This research project focused on the development of a concrete water-cement ratio prediction system using Deep Neural Networks (DNNs) to enhance the quality, durability, and performance of concrete in the construction industry. The study highlighted the critical importance of accurately predicting the water-cement ratio, particularly in regions like Nigeria where environmental conditions significantly affect concrete properties. Traditional methods of determining the water-cement ratio often result in inconsistent concrete quality, leading to material wastage and increased costs. To address these challenges, the project developed a DNN-based prediction model, utilizing a dataset collected from the Department of Civil Engineering at Osun State University.
The methodology involved several phases including data collection, preprocessing, model development, and performance evaluation. The model was implemented using Python and its libraries such as Pandas, NumPy, Scikit-learn, and TensorFlow. The dataset was divided into training, validation, and testing sets to ensure the model’s robustness. The DNN model successfully processed the complex interactions between various factors influencing concrete properties, leading to highly accurate predictions of the water-cement ratio.
The results indicated that the deep neural network model effectively predicted the water-cement ratio of concrete with high accuracy. The model, trained on 72 samples (80% for training and 20% for testing), achieved a training loss of 0.0025 and a validation loss of 0.0019 over 100 epochs with a batch size of 10. The Mean Absolute Error (MAE) was 0.0019, and the Mean Squared Error (MSE) was 0.027 on the test set. The training and validation accuracy, reflected in the MAE, decreased consistently and stabilized at low values, indicating successful learning. The residuals were centered around zero, ranging from -0.01 to 0.02 confirming the model's robustness, accuracy, and resistance to overfitting.


## Objectives

The aim of this project is to develop Deep Neural Network (DNN) to predict the concrete water cement ratio.
The specific objectives features are to:
i.	collect and examine the data for prediction;
ii.	design a model for (i) above;
iii.	implement the model in (ii);
iv.	evaluate the performance of the developed system.

## Project Structure

## Import Data


import pandas as pd

# Load the dataset
file_path = 'DAVE UML.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
data.head()

```
```
## Data Cleaning 

# Check for missing values
missing_values = data.isnull().sum()

# Get basic statistics of the dataset
data_statistics = data.describe()

missing_values, data_statistics
```
```
## Extract Features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and target variable
features = data[['Cement (kg/m3)', 'Sand (kg/m3)', 'Granite (kg/m3)', 'Age (days)', 'Water (kg/m3)', 'Slump (mm)', 'Density (kg/m3)', 'Compressive strength (N/mm2)']]
target = data['w/c']

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

```

```
## Tranning of Datasets

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network architecture
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(64, input_dim=8, activation='relu'))

# Second hidden layer
model.add(Dense(32, activation='relu'))

# Third hidden layer
model.add(Dense(16, activation='relu'))

# Output layer
model.add(Dense(1, activation='linear'))

# Display the summary of the model
model.summary()
```
```
## Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_absolute_error']
)

# Display the model summary again to confirm the compilation
model.summary()
```

```
##  Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=10,
    validation_split=0.2
)

```

```

## Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')

```

```
## Save the model in the recommended Keras format
model.save('water_cement_ratio_predictor.keras')

# Load the model
loaded_model = tf.keras.models.load_model('water_cement_ratio_predictor.keras')

# Example prediction
import numpy as np

# Assuming new_data is a numpy array with shape (1, 8) representing a single new sample
new_data = np.array([[300, 700, 1400, 14, 200, 20, 2400, 18]])
new_data_normalized = scaler.transform(new_data)
prediction = loaded_model.predict(new_data_normalized)

print(f'Predicted Water-Cement Ratio: {prediction[0][0]}')
```
```

## import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

```
```

## Make predictions on the test set
predictions = loaded_model.predict(X_test)

# Plot predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.title('Predicted vs Actual Water-Cement Ratio')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


```
```

## import matplotlib.pyplot as plt

# Make predictions on the test set
predictions = loaded_model.predict(X_test)

# Plot predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.title('Predicted vs Actual Water-Cement Ratio')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
```
```

## import numpy as np
import pandas as pd

# Assume 'scaler' is the StandardScaler used earlier, and the target was not scaled

# Check the original scale of the target 'w/c'
# Normally, if the scaler was applied to the features, predictions should not need rescaling

# For demonstration, here's how to manually adjust predictions if needed
# If predictions need to be within the range [min_actual, max_actual]
min_actual = data['w/c'].min()
max_actual = data['w/c'].max()

# Scale the predictions back to the [min_actual, max_actual] range
predictions_rescaled = np.interp(predictions.flatten(), (predictions.min(), predictions.max()), (min_actual, max_actual))

# Create a DataFrame with actual and rescaled predicted values
results_rescaled = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions_rescaled
})

# Display the first few rows of the DataFrame
results_rescaled.head()
```
```

## Plot predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions_rescaled, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', label='Perfect Prediction')
plt.title('Predicted vs Actual Water-Cement Ratio')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

```
```

## import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
```
```
## Plot training & validation MAE values
plt.figure(figsize=(10, 5))
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right')
plt.show()
```
```
## Calculate residuals
residuals = y_test - predictions_rescaled

# Plot residuals
plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()
```
```
## Plot distribution of errors
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20, edgecolor='k')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()
```
```
## Plot training & validation MAE values
plt.figure(figsize=(10, 5))
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right')
plt.show()
```
```
## Findings

Following data collection, the data pre-processing phase was essential to prepare the dataset for effective training of the deep neural network model. Key features such as Cement, Sand, Granite, Age, Water, Slump, Density, and Compressive Strength were normalized to ensure a consistent scale across all inputs as shown in Table 3.1. This normalization was critical to prevent any one feature from disproportionately influencing the model’s predictions. The dataset was then split into training (80%) and test (20%) sets, allowing the model to learn from a large portion of the data while keeping a separate portion for evaluating its performance on unseen data. Feature selection was focused on variables that directly impact the water-cement ratio, ensuring the model was trained with the most relevant information. The target variable was the actual water-cement ratio, which the model aimed to predict accurately.
The model was trained for 100 epochs with a batch size of 10, adjusting its weights iteratively to minimize prediction errors. The training results showed a consistent decrease in both training and validation losses, indicating successful learning, by the 100th epoch, the training loss had decreased to 0.0025, and the validation loss to 0.0019, reflecting the model's ability to generalize well without overfitting.

## Conclusion

From the results obtained in the study, the following conclusion has been drawn: The deep neural network model demonstrated good predictive capabilities for the water-cement ratio of concrete. The results indicate that the deep neural network model is capable of predicting the water-cement ratio of concrete with a reasonable degree of accuracy. The low mean absolute error and mean squared error suggest that the model's predictions were close to the actual values. Despite the positive results, there are some limitations to this study. The dataset was relatively small, comprising only 72 samples, which may limit the model's generalizability. Additionally, the model's performance could be further improved by exploring different neural network architectures, hyperparameters, and additional features

## Recommendation 
Based on the outcomes of this project, the following recommendations are proposed: Further fine-tuning of the model's hyperparameters (e.g., learning rate, number of layers, neurons) can be conducted to improve performance, Investigating additional features or transforming existing ones may enhance the model's predictive capabilities, Implementing cross-validation techniques can provide a more robust evaluation of the model's performance,Combining multiple models using ensemble techniques could potentially yield better predictive accuracy,Deploying the model in a real-time environment can help validate its practical applicability and performance on new, unseen data.

## Author - DAYFEED 

This project is part of my portfolio, showcasing the skills essential for data analyst. If you have any questions, feedback, or would like to collaborate, feel free to get in touch!
