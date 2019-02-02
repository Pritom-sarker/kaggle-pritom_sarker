#prepare dataset


import pandas as pd
train=pd.read_csv("titanic/train.csv")
#print(train.head())
numeric=['Pclass','Age','SibSp','Parch','Fare']
x_train=train[numeric]
y_train=train['Survived']
x=y_train.values
#print(x_train.head())
# print(y_train.head())
#
#
#
# print(x_train["Age"].isnull().values.any()) # Check that have any null value or not
#
# md=x_train["Age"].median()
# x_train["Age"].fillna(md)

# Before filling the empty value


print(x_train["Age"].isnull().values.any()) # Check that have any null value or not
#Imputer Fill all the nan value with Median

from  sklearn.preprocessing import Imputer

imput=Imputer(strategy="median")
imput.fit(x_train)
x=imput.transform(x_train)

final_train_X=pd.DataFrame(x,columns=x_train.columns)

#After fill the nan value with median using

print(final_train_X["Age"].isnull().values.any()) # Check that have any null value or not


# Train classifire

from sklearn.linear_model import SGDClassifier

sgd=SGDClassifier()
sgd.fit(final_train_X,y_train)

#predict using classifire

print(sgd.predict([[3,22.0,1 ,0,7.2500]]))  # USe a 2D arrey to check

# calculate cost'with k fold cross validation


from sklearn.model_selection import cross_val_score
#print(cross_val_score(sgd, final_train_X, y_train, cv=4, scoring="accuracy").mean()) #it returns a 1D arry . we take the mean of that array



#Check answer and saved to dataset

#Load test data
test=pd.read_csv("titanic/test.csv")
numeric=['Pclass','Age','SibSp','Parch','Fare']
x_test=test[numeric]
x_test=x_test.reindex()

# Check there any null value or not then replace using imputer class

#print(x_test.isnull().values.any())
imput.fit(x_test)
test_x=imput.transform(x_test)
test_x=pd.DataFrame(test_x,columns=x_test.columns)
#print(test_x.isnull().values.any())


# Predict using classifire

pred=sgd.predict(test_x)
#print(pred)
data=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred})
data.to_csv("answer.csv")
print("Data saved !!")