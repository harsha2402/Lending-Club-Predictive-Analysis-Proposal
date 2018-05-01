
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#first file rejections file 
file="C:/Users/harsh/Downloads/Loans/RejectStatsD.csv"
file1="C:/Users/harsh/Downloads/Loans/LoanStats3d.csv"
df =(pd.read_csv(file))
final_data=df.sample(n=10000, replace=False)

#dropping the risk score from rejections file as it will be included into the final project
final_data.drop('Risk_Score', axis=1, inplace=True)

#preparing the data
final_data.rename(index=str,columns={"Amount Requested":"loan_amnt","Loan Title":"title","Debt-To-Income Ratio":"dti","Zip Code":"zip_code","State":"addr_state","Employment Length":"emp_length","Policy Code":"policy_code"}, inplace=True)
dti=pd.to_numeric(final_data['dti'].str.replace('%',''))
final_data.drop(['dti'],axis=1,inplace=True)
final_data['dti']=dti

#second file accepted loans file with filtered data according to rejections data
df =(pd.read_csv(file1))
df1=df.sample(n=10000,replace=False)
df2=df1[['loan_amnt','title','dti','zip_code','addr_state','emp_length','policy_code']]
frames=[final_data[:10000],df2[0:10000]]
data=pd.concat(frames)

#converting nominal data to numerical data
ny=[]
for row in data['addr_state']:
    if row=='NY':
        ny.append(1)
    else :
        ny.append(0)
data['NY']=ny
data.drop(['addr_state'],axis=1,inplace=True)
data.drop(['zip_code'],axis=1,inplace=True)
lessthanyr=[]
for row in data['emp_length']:
    if row=='< 1 year':
        lessthanyr.append(1)
    else :
        lessthanyr.append(0)
data['Less_than_1yr']=lessthanyr
data.drop(['emp_length'],axis=1,inplace=True)
debt=[]
for row in data['title']:
    if row=='Debt consolidation':
        debt.append(1)
    else :
        debt.append(0)
data['Debt_type']=lessthanyr
data.drop(['title'],axis=1,inplace=True)

data=data.sample(frac=1)
test=data[0:2000]
train=data[2000:]

#Fitting the data into logistic regression. 
#***the policy code is used as the label here policy_code=0 meaning the loan was rejected
logisticRegr = LogisticRegression()
logisticRegr.fit(train[['dti','loan_amnt','NY','Less_than_1yr','Debt_type']], train['policy_code'])
predicted_code=(logisticRegr.predict(test[['dti','loan_amnt','NY','Less_than_1yr','Debt_type']]))
accuracy = logisticRegr.score(test[['dti','loan_amnt','NY','Less_than_1yr','Debt_type']], test['policy_code'])
print("the accuracy of the Logistic regression model is: ",accuracy)


# In[ ]:



