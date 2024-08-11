import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
credited_card_df=pd.read_csv('.venv/creditcard.csv')
legit = credited_card_df[credited_card_df.Class==0]

fraud=credited_card_df[credited_card_df['Class'] ==1 ]
legit_sample = legit.sample(n=492)

credited_card_df=pd.concat([legit_sample,fraud],axis=0)
X = credited_card_df.drop('Class',axis=1)
Y = credited_card_df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_scaled, Y_train)

train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

#webapp

st.title("Credit Card  Fraud Detection Model ")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")


input_df = st.text_input('Input All features')
input_df_lst=input_df.split(',')
#submit button
submit = st.button("Submit")


if submit :
    features= np.array(input_df_lst,dtype=np.float64)
    #make prediction
    prediction=model.predict(features.reshape(1,-1))
    #display result
    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulent Transaction")

