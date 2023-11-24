import numpy as np
import pickle
import pandas as pd
from xgboost import XGBRegressor

# loading the saved model
loaded_model = pickle.load(open('C:/Users/User/PycharmProjects/MedicalCostPredictionML/insurancemodelf.pkl', 'rb'))

# Define the input data
input_data = [19, 1, 27.900, 0, 1, 3]

# Create a DataFrame with the input data
input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Select the relevant 4 features used during training
input_selected_features = input_df[['age', 'bmi', 'children', 'smoker']]

# Make predictions on the new data
predictions = loaded_model.predict(input_selected_features)

# Display the predictions
print(predictions)
print('The insurance cost in USD ',predictions[0])