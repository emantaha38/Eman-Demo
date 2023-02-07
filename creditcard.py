import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np

st.title("Credit Card Score APP")
        
# Take inputs from user
credit_Mix = st.selectbox('Credit_Mix',['Standard', 'Good', 'Bad'])
payment_of_Min_Amount = st.selectbox('Payment_of_Min_Amount', ['No', 'Yes'])
payment_Behaviour = st.selectbox('Payment_Behaviour', ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments','Low_spent_Small_value_payments','High_spent_Medium_value_payments','High_spent_Large_value_payments','Low_spent_Medium_value_payments'])
age_Group = st.selectbox('Age_Group', ['14-25', '25-30','30-45','45-55','55-70','70-100'])
outstanding_Debt = st.slider("Outstanding_Debt", 0, 5000)
monthly_Inhand_Salary = st.slider("Monthly_Inhand_Salary", 300, 15000)
annual_Income = st.slider("Annual_Income", 7000, 200000)
total_EMI_per_month = st.slider("Total_EMI_per_month", 0, 2000)
changed_Credit_Limit = st.slider("Changed_Credit_Limit", 0,40) 
delay_from_due_date = st.slider("Delay_from_due_date", 0,70)
interest_Rate = st.slider("Interest_Rate", 1,60)
num_Bank_Accounts = st.slider("Num_Bank_Accounts", 1,10)
num_Credit_Card = st.slider("Num_Credit_Card", 0,10)
num_of_Loan = st.slider("Num_of_Loan", 0,10)
num_of_Delayed_Payment = st.slider("Num_of_Delayed_Payment", 1,20)
num_Credit_Inquiries = st.slider("Num_Credit_Inquiries", 0,10)
credit_Utilization_Ratio = st.slider("Credit_Utilization_Ratio", 20,50)
credit_History_Age = st.slider("Credit_History_Age", 1,30)
amount_invested_monthly = st.slider("Amount_invested_monthly", 0,2000)
monthly_Balance = st.slider("Monthly_Balance", 0,2000)
         
# Convert inputs to DataFrame
         
train_new = pd.DataFrame({"Outstanding_Debt":[outstanding_Debt],"Monthly_Inhand_Salary":[monthly_Inhand_Salary],"Annual_Income":[annual_Income],"Total_EMI_per_month":[total_EMI_per_month],"Changed_Credit_Limit":[changed_Credit_Limit],"Delay_from_due_date":[delay_from_due_date],"Interest_Rate":[interest_Rate],"Num_Bank_Accounts":[num_Bank_Accounts],"Num_Credit_Card":[num_Credit_Card],"Num_of_Loan":[num_of_Loan],"Num_of_Delayed_Payment":[num_of_Delayed_Payment],"Num_Credit_Inquiries":[num_Credit_Inquiries],"Credit_Utilization_Ratio":[credit_Utilization_Ratio],"Credit_History_Age":[credit_History_Age ],"Amount_invested_monthly":[amount_invested_monthly],"Monthly_Balance":[monthly_Balance],"Credit_Mix":[credit_Mix],"Payment_of_Min_Amount":[payment_of_Min_Amount],"Payment_Behaviour":[payment_Behaviour],"Age_Group":[age_Group]})  
         
# Load the transformer
         

trans=pkl.load(open(r'C:\Users\Yazeed\Downloads\transformer.pkl', 'rb'))
# Apply the transformer on the inputs

X_new = trans.transform(train_new) 

         
# Load the model

loaded_model=pkl.load(open(r'C:\Users\Yazeed\Downloads\XGboot.pkl','rb')) 
         
# Predict the output
credit_card_score_pre = loaded_model.predict(X_new)[0]
if st.button("Predict"): 
    if(credit_card_score_pre==0):
        st.write("Good")
    elif(credit_card_score_pre==1):
        st.write("Poor")
    else:
        st.write("Standard")
    
    