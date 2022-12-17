import pickle as pkl
import streamlit as st
import pandas as pd


# Take inputs from user
st.image('Income.jpg')
education = st.selectbox('Education', ['Pre-mid-school', 'HS-grad', 'Assoc', 'Bachelors', 'Masters', 'Prof-degree'])
work_type = st.selectbox('Work Type (hrs)', ['< 32', '32-40', '41-72', '> 72'])
age = st.selectbox('Age', ['< 30', '30-39','40-49','50-60','> 60'])
capital = st.selectbox('Capital', ['Loss-above-1.9k','Loss-below-1.9k','No-loss-gain','Gain-below-7K','Gain-above-7K'])
marital_status = st.selectbox('marital-status', ['Married', 'Single', 'Not-married'])
relationship = st.selectbox('Relationship', ['Partner', 'Not-in-family', 'Own-child', 'Other-relative'])
occupation = st.selectbox('Occupation', ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Adm-clerical', 'Machine-op-inspct', 'Transport-moving', 'Tech-support', 'Farming-fishing', 'Protective-serv ', 'Other-service'])
gender = st.selectbox('Gender', ['Male', 'Female'])

# Convert inputs to DataFrame
df_new = pd.DataFrame({'education': [education], 'work-type': [work_type], 'age-range': [age], 'capital-remain-range':[capital],'marital-status': [marital_status], 'relationship': [relationship], 'occupation': [occupation], 'gender': [gender]})

# Load the transformer
transformer = pkl.load(open('transformer_2.pkl', 'rb'))

# Apply the transformer on the inputs
X_new = transformer.transform(df_new)

# Load the model
model = pkl.load(open('optimized_svc_2.pkl', 'rb'))

# Predict the output
predict = model.predict(X_new)
# prop = model.predict_proba(X_new)[0][1] * 100

dic = {0: '<=50K', 1:'>50K'}
st.markdown('## Income is : '+str(dic[predict[0]]))
# st.markdown(f'## With Probability : {round(prop, 2)} %')

