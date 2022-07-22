import pandas as pd
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
b = pd.read_csv("C:\\Users\\tvste\\OneDrive\\Desktop\\HR_comma_sep.csv")
b1 = b[b['left']==1]
b2 = b[b['left']==0]
c= pd.crosstab(b.Department,b.left)
d = pd.pivot_table(b,index = ['Department'],columns=['left'], values=['satisfaction_level'])
df = b[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(d)
salary_dummies = pd.get_dummies(df['salary'],prefix = 'salary')
df1 = pd.concat([df,salary_dummies],axis = 'columns')
df2 = df1.drop(['salary'],axis='columns')
from sklearn.model_selection import train_test_split
y = b['left']
x_train,x_test,y_train,y_test = train_test_split(df2,y,test_size=0.3)
z = linear_model.LogisticRegression()
z.fit(x_train,y_train)
print(x_test)
print(z.score(x_test,y_test))
