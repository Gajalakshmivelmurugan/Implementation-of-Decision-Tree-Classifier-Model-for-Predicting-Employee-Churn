# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1:start the program

step 2:import pandas module and import the required data set.

step 3:Find the null values and count them.

step 4:Count number of left values.

step 5:From sklearn import LabelEncoder to convert string values to numerical values.

step 6:From sklearn.model_selection import train_test_split.

step 7:Assign the train dataset and test dataset.

step 8:From sklearn.tree import DecisionTreeClassifier.

step 9:Use criteria as entropy.

step 10:From sklearn import metrics.

step 11:Find the accuracy of our model and predict the require values.

step 12:End the program

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: gajalakshmi V
RegisterNumber:212223040047  
*/
import pandas as pd
data = pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
Data:
![369327006-28c5a398-7f32-4b79-acd0-ec36a2c76c60](https://github.com/user-attachments/assets/e9781bb2-f758-40fb-b45b-50e809f1893b)

Accuracy:

![369327110-b9b1dd85-7b53-4299-beda-b04dd0b7b9d3](https://github.com/user-attachments/assets/1681e31a-c1e9-4772-b5a8-df9b39fe5a90)

Predict:

![369327349-1b084a37-90af-4668-8285-51097e9ec346](https://github.com/user-attachments/assets/5ae05d12-542e-44a9-a340-dcd187318286)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
