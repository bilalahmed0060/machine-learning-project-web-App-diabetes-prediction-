import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,auc,roc_auc_score

#load data and extract independent and dependent variable

df = pd.read_csv('diabetes.csv')
print(df['BloodPressure'])
print("=======shape=======")
print(df.shape)
print("======info======")
print(df.info())
print('null values')
print(df.isnull().sum())

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
y = np.array(y)
y = y.ravel()

#data visualization
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
rr = [X['Pregnancies'],X['Glucose'],X['BloodPressure'],X['SkinThickness'],X['Insulin'],X['BMI'],X['DiabetesPedigreeFunction'],X['Age']]

figer,ax = plt.subplots(2, 4,figsize=(12, 8))

ax[0,0].hist(X['Pregnancies'],bins=35,color='green')
ax[0,0].set_title('Pregnancies')


ax[0,1].hist(X['Glucose'],bins=35,color='green')
ax[0,1].set_title('Glucose')

ax[0,2].hist(X['BloodPressure'],bins=35,color='green')
ax[0,2].set_title('BloodPressure')

ax[0,3].hist(X['SkinThickness'],bins=35,color='green')
ax[0,3].set_title('SkinThickness')

ax[1,0].hist(X['Insulin'],bins=35,color='green')
ax[1,0].set_title('Insulin')

ax[1,1].hist(X['BMI'],bins=35,color='green')
ax[1,1].set_title('BMI')

ax[1,2].hist(X['DiabetesPedigreeFunction'],bins=35,color='green')
ax[1,2].set_title('DiabetesPedigreeFunction')

ax[1,3].hist(X['Age'],bins=35,color='green')
ax[1,3].set_title('Age')
plt.show()

#Avoiding Dummy Variable
X.dropna()
X['BloodPressure'].loc[(X['BloodPressure'] == 0)] = df['BloodPressure'].median()
#Splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print('y test\n', y_test)
#fitting the model to training the set

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn = knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
knnn = prediction
print("====================Prediction Of model(KNN)=================")
print(prediction)
print("====================ACtual Answers=================")
print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=knn.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
print(roc_auc_score(y_test,prediction))

from sklearn.naive_bayes import GaussianNB

nf = GaussianNB()
nf.fit(X_train, y_train)
prediction = nf.predict(X_test)
nff = prediction
print("====================Prediction Of model(Naive Bayes)=================")
print(prediction)
print("====================ACtual Answers=================")
print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=nf.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

lr=LogisticRegression()
modelLR=lr.fit(X_train,y_train)

#Predicting the test set results
prediction = modelLR.predict(X_test)
lrr = prediction
print("===============Prediction Of model(logistic regression)=================")
print(prediction)
print("==================ACtual Answers===================")
print(y_test)
print("=====================Training Accuarcy=============")
trac=lr.score(X_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
prediction = dt.predict(X_test)
dtt = prediction
print("====================Prediction Of model(Decision Tree)=================")
print(prediction)
print("====================ACtual Answers=================")
print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=knn.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
rfc = prediction
print("====================Prediction Of model(Random Forest)=================")
print(prediction)
print("====================ACtual Answers=================")

print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=rf.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))


from sklearn.svm import SVC

sv = SVC()
sv.fit(X_train, y_train)
prediction = sv.predict(X_test)
svc = prediction
print("====================Prediction Of model(SVC)=================")
print(prediction)
print("====================ACtual Answers=================")
print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=sv.score(X_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction)
testingAcc=teac*100
print(testingAcc)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
# print("========array========")
# act = np.array(y_test)
# act = act.flatten()
# print(act)

classifier = {
       'KNN' : knnn,
       'NB'  : nff,
       'LR'  : lrr,
       'RF'  : rfc,
       'DT'  : dtt,
       'SVC' : svc,
       'Y-TEST': y_test
       }

df1 = pd.DataFrame(classifier, columns = ['KNN', 'NB','LR','RF','DT','SVC','Y-TEST'])
df1.to_excel (r'C:/Users/CCS LAPTOP HYD/Desktop/prediction_dataframe.xlsx', index = False, header=True)


