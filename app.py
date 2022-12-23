import pickle as pkl
import streamlit as st
import pandas as pd

# Take inputs from user
st.image('Income.jpg')
education = st.selectbox('Education', 
                         ['Pre-Middle School', 'High School Grade', 'Associate Degree', 'Bachelors Degree', 
                          'Masters Degree', 'Doctorate / Professional Degree'])
age = st.slider("Age", 17, 90)
hrs_per_week = st.slider("Working Hours Per Week", 1, 99)
capital_gain = st.slider("Capital Gain", 0, 99999)
capital_loss = st.slider("Capital Loss", 0, 99999)
relationship = st.selectbox('Relationship', ['Partner', 'Not-in-family', 'Own-child', 'Other-relative'])
marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Not-married'])
occupation = st.selectbox('Occupation', ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Adm-clerical',
                                         'Machine-op-inspct', 'Transport-moving', 'Tech-support',
                                         'Farming-fishing', 'Protective-serv', 'Other-service'])
gender = st.selectbox('Gender', ['Male', 'Female'])

# Capital Mapping 
capital = '0'
diff = (capital_gain - capital_loss)
if diff <= -1900:
    capital = '<= -1900'
elif -1900 < diff < 0:
    capital = '> -1900'
elif 0 < diff <= 7000:
    capital = '<= 7000'
elif diff > 7000:
    capital = '> 7K'
   
# Education Mapping 
edu_dic = {
'Pre-Middle School': 'Pre-mid-school',
'High School Grade': 'HS-grad',
'Associate Degree': 'Assoc',
'Bachelors Degree': 'Bachelors',
'Masters Degree': 'Masters',
'Doctorate / Professional Degree': 'Prof-degree',
}

# Convert inputs to DataFrame
df_new = pd.DataFrame({'age': [age],
                       'hours-per-week': [hrs_per_week], 
                       'education': [edu_dic[education]],
                       'capital-remain-range':[capital],
                       'marital-status': [marital_status], 
                       'occupation': [occupation], 
                       'relationship': [relationship], 
                       'gender': [gender]})

# Load the transformer
transformer = pkl.load(open('transformer.pkl', 'rb'))

# Apply the transformer on the inputs
X = transformer.transform(df_new)

# Load the model
model = pkl.load(open('voting_eclf.pkl', 'rb'))

# Predict the output
predict = model.predict(X)
prop = model.predict_proba(X)[0][1] * 100

dic = {0: '<=50K', 1:'>50K'}
st.markdown('## Income : '+str(dic[predict[0]]))
st.markdown(f'## Probability : {round(prop, 2)} %')

