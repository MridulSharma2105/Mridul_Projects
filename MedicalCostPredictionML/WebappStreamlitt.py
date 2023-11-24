import string

import numpy as np
import pickle
import pandas as pd
from xgboost import XGBRegressor
import streamlit as st

# loading the saved model
#loaded_model = pickle.load(open('E:/Medial Insurance Cost Prediction/costtrained_model.sav', 'rb'))
loaded_model = pickle.load(open('C:/Users/User/PycharmProjects/MedicalCostPredictionML/insurancemodelf.pkl', 'rb'))


# creating a function for prediction
def medical_cost_prediction(input_data):
    # changing input_data to numpy array
    input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Select the relevant 4 features used during training
    input_selected_features = input_df[['age', 'bmi', 'children', 'smoker']]

    # Make predictions on the new data
    predictions = loaded_model.predict(input_selected_features)

    # Display the predictions
    print(predictions)
    final_ans = 'The insurance cost in USD ', predictions[0]
    return final_ans


def main():
    # giving a title
    st.title('Medical Cost Prediction Web App')

    # getting the data from the user
    age = st.text_input('Enter your age')
    sex = st.selectbox('Gender', ('male', 'female'))
    bmi = st.text_input('Enter your BMI value')
    children = st.text_input('No. of children')
    smoker = st.selectbox('Do you smoke?', ('yes', 'no'))
    region = st.selectbox('Which region are you from?', ('southeast', 'southwest', 'northeast', 'northwest'))

    # necessary conversions

    # sex = sex.replace("male", "0")
    # try:
    #     sex = sex.replace("female", "1")
    #
    # except ValueError:
    #     sex = 1
    #
    # sex = int(sex)
    if sex == "male":
        sex = 0
    else:
        sex = 1

    smoker = smoker.replace("no", "0")
    smoker = smoker.replace("yes", "1")
    smoker = int(smoker)

    region = region.replace("southeast", "0")
    region = region.replace("southwest", "1")
    region = region.replace("northeast", "2")
    region = region.replace("northwest", "3")
    region = int(region)

    # code for Prediction
    cost = ''

    # creating a button for Prediction

    if st.button('Medical Cost Result'):
        a = [age, sex, bmi, children, smoker, region]
        b = np.array(a, dtype=float)  # convert using numpy
        cost = medical_cost_prediction(b)

    st.success(cost)


if __name__ == '__main__':
    main()
