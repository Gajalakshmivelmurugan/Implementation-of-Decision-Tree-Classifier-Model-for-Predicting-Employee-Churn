# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: gajalakshmi V
RegisterNumber:212223040047  
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![322719919-881c91dd-e1b0-43c4-8e8c-c02819e7af2f](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/f5a809b7-d3d8-43a0-bf52-8adf210bfb67)
![322720014-048f14a5-56cd-4879-8696-c87db2bff1ff](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/218905e6-7618-4521-86df-4dfed8fa65ad)
![322720106-e81e418a-c91b-41e2-a0da-ba09836993c2](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/8057ad97-2c3a-4e71-bf87-0d752282c59a)
![322720243-0b10b933-f019-4c1b-888a-3a0d86b767e3](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/edc20296-1d74-4007-9f0b-260b6925bd18)
![322720339-24c48dc3-7c7e-4162-84e4-d97b69bd0569](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/a12c2493-2dd3-43e9-b755-39f1df42aeb9)
![322720480-10d1a7aa-68ff-434e-9fd3-cb035be62323](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/b5f60326-e787-4b32-ba2c-ab2382994c9e)
![322720580-5c88de7e-879f-4c53-b2d3-780a8ac6a7d5](https://github.com/Gajalakshmivelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871940/13a6f482-87f3-43ea-aa49-521709863a99)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
