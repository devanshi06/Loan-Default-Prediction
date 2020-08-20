import pandas as pd
import numpy as np

bank=pd.read_csv("D:\\Project\\bank_final.csv")
print(bank.shape)

#Preprocessing of the data
bank.isnull()
print(bank.dtypes)
print(bank.info())
print(bank.describe())

dict={}
for col in bank:
    dict[col]=bank[col].value_counts()
dict

values={'Name':"SUBWAY",'City':"LOS ANGELES",'State':"CA",'Zip':10001,'BankState':"NC",'Bank':"BANK OF AMERICA NATL ASSOC",'CCSC':0, 'ApprovalDate':"30-Sep-97",'ApprovalFY':2006,'Term':84,'NoEmp':1,'NewExist':1,'CreateJob':0,'RetainedJob':0,'FranchiseCode':1,'UrbanRural':1,'RevLineCr':"N",'RevLineCr':"N",'LowDoc':"N",'ChgOffDate':0,'DisbursementDate':"31-May-06",'DisbursementGross':50000.00,'BalanceGross':0.00,'MIS_Status':"P I F",'ChgOffPrinGr':0.00,'GrAppv':50000.00,'SBA_Appv': 25000.00}
bank1=bank.fillna(value=values)
bank1.info()
bank1['DisbursementGross']=bank1['DisbursementGross'].str.replace('\W','').astype(float)
bank1['BalanceGross']=bank1['BalanceGross'].str.replace('\W','').astype(float)
bank1['ChgOffPrinGr']=bank1['ChgOffPrinGr'].str.replace('\W','').astype(float)
bank1['GrAppv']=bank1['GrAppv'].str.replace('\W','').astype(float)
bank1['SBA_Appv']=bank1['SBA_Appv'].str.replace('\W','').astype(float)
print(bank1)

#Data Cleaning performed

#EDA
#MIS Status has PIF 110123 and CHGOFF is 39008
#Converting MIS status to numerical

target_cat=bank1['MIS_Status']
target_num=[]

for label in target_cat:
    label=str(label).strip()
    if label == "CHGOFF":
        target_num.append(0)
    else:
        target_num.append(1)
    
        
print(len(target_cat))
print(len(target_num))

print(target_cat[:10])
print(target_num[:10])

t_cat=bank1['RevLineCr']
t_num=[]

for i in t_cat:
    i=str(i).strip()
    if i == "N":
        t_num.append(0)
    else:
        t_num.append(1)

print(len(t_cat))
print(len(t_num))
print(t_cat[:35])
print(t_num[:35])

cat=bank1['LowDoc']
num=[]

for j in cat:
    j=str(j).strip()
    if j == "N":
        num.append(0)
    else:
        num.append(1)
        
print(len(cat))
print(len(num))
print(cat[:35])
print(num[:35])


#Assign target in data
bank1["target_Num"]=target_num
bank1["RevLineCr"]=t_num
bank1["LowDoc"]=num
bank1.head(10)
print(bank1.corr())

import seaborn as sns
sns.boxplot(data=bank1,orient="h",palette="Set2")
sns.pairplot(bank1)
sns.heatmap(bank1.corr())

import matplotlib.pyplot as plt

sns.countplot(bank1["ApprovalDate"],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])
# Not having much impact on the target as distribution is small

sns.countplot(bank1['ApprovalFY'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=1.0)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])

sns.countplot(bank1['State'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])


sns.countplot(bank1['Bank'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])
# Not having much impact on the target as distribution is small

sns.countplot(bank1['City'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])
# Not having much impact on the target as distribution is small

sns.countplot(bank1['Zip'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])
# Not having much impact on the target as distribution is small

sns.countplot(bank1['CCSC'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])

sns.countplot(bank1['DisbursementDate'],data=bank1,hue='target_Num')
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])
# Not having much impact on the target as distribution is small


#Similarly using crosstab  to find potential defaulters
pd.crosstab(bank1.ApprovalDate,bank1.target_Num,normalize='index')
pd.crosstab(bank1.ApprovalFY,bank1.target_Num,normalize='index')
pd.crosstab(bank1.State,bank1.target_Num,normalize='index')
pd.crosstab(bank1.Bank,bank1.target_Num,normalize='index')
pd.crosstab(bank1.City,bank1.target_Num,normalize='index')
pd.crosstab(bank1.Zip,bank1.target_Num,normalize='index')
pd.crosstab(bank1.CCSC,bank1.target_Num,normalize='index')
pd.crosstab(bank1.MIS_Status,bank1.target_Num,normalize='index')
pd.crosstab(bank1.DisbursementDate,bank1.target_Num,normalize='index')

banking=bank1.drop(["Zip","CCSC","City","Bank","ApprovalDate","DisbursementGross","GrAppv","ChgOffDate","BankState","Name","DisbursementDate","MIS_Status"],axis=1)
sns.boxplot(data=banking,orient="h",palette="Set2")
banking

#dummy variables for categorical column

categorical_column=['State']
from sklearn import preprocessing
for i in categorical_column:
    num=preprocessing.LabelEncoder()
    banking[i]=num.fit_transform(banking[i])
print(banking.head(10))
input()
#bank_dummy=pd.get_dummies(banking,columns=categorical_column)


#Down sizing samples

from sklearn.utils import resample
bank_majority=banking[banking.target_Num==1]
bank_minority=banking[banking.target_Num==0]

bank_majority
bank_minority

majority_down=resample(bank_majority,replace=False,n_samples=7000,random_state=123)
minority_down=resample(bank_minority,replace=False,n_samples=5000,random_state=123)

majority_down
minority_down

banking_sample=pd.concat([majority_down,minority_down])
banking_sample=pd.concat([bank_majority,bank_minority])
X=banking_sample.drop('target_Num',axis=1)
Y=banking_sample[["target_Num"]]


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

#KNN based modeling
from sklearn.neighbors import KNeighborsClassifier as KNN
neigh=KNN(n_neighbors=2)

neigh.fit(X_train,Y_train)
prediction=neigh.predict(X_train)
prediction=pd.DataFrame(prediction)
train_acc=np.mean(prediction.values==Y_train.values)
print(train_acc)
#Train Accuracy 99.12%

test_acc=np.mean((pd.DataFrame(neigh.predict(X_test))).values==Y_test.values)
print(test_acc)
#Test Accuracy 98.6%

#Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

pred_gnb=GaussianNB().fit(X_train,Y_train).predict(X_test)
pred_mnb=MultinomialNB().fit(X_train,Y_train).predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,pred_gnb)
pd.crosstab(Y_test.values.flatten(),pred_gnb)
np.mean(pred_gnb==Y_test.values.flatten())

confusion_matrix(Y_test,pred_mnb)
pd.crosstab(Y_test.values.flatten(),pred_mnb)
np.mean(pred_mnb==Y_test.values.flatten())
#Test Accuracy 98.6%

#Decision tree
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
prd_dt=dt.predict(X_test)
pd.Series(prd_dt).value_counts
np.mean(Y_train.values==pd.DataFrame(dt.predict(X_train)).values)
#Train Accuracy 99.99
np.mean(Y_test.values==pd.DataFrame(prd_dt).values)
#Test Accuracy is 98.1

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=200,criterion="entropy")

rf.fit(X_train,Y_train)
rf.estimators_
rf.classes_
rf.n_classes_
rf.n_features_
rf.n_outputs_
rf.oob_score_
rf.predict(X_train)

predict=rf.predict(X_train)
confusion_matrix(Y_train,predict)
print("Accuracy",(31206+88790)/(31206+88792+3+0)*100)
# Accuracy of train data is 99.99%

predict_test=rf.predict(X_test)
confusion_matrix(Y_test,pd.DataFrame(predict_test).values)
#


#Logistic Regression
#from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn import preprocessing
X_train_s=preprocessing.scale(X_train)
X_test_s=preprocessing.scale(X_test)

from sklearn.linear_model import LogisticRegression

Logreg=LogisticRegression(solver='lbfgs')
Logreg.fit(X_train_s,Y_train)

##prediction for test set
y_prd=Logreg.predict(X_test_s)
y_trainpred=Logreg.predict(X_train_s)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
#cross_validation=model_selection.cross.val_score(Logreg,X_train_s,Y_train,cv=kfold,score)

cls_report=classification_report(Y_test,y_prd,output_dict=True)
print(cls_report)
#test Accuracy 97.75%

Lg_trainreport=classification_report(Y_train,y_trainpred)
print(Lg_trainreport)
#Train Accuracy 98%

#SVM based model
from sklearn.svm import SVC

linear=SVC(kernel="linear")
linear.fit(X_train_s,Y_train)
pred_linear_test=linear.predict(X_test_s)
confusion_matrix(pred_linear_test,Y_test)
svmlinear_testreport=classification_report(pred_linear_test,Y_test)
print(svmlinear_testreport)
#Test Accuracy is 98%

svmlinear_trainreport=classification_report(linear.predict(X_train_s),Y_train)
print(svmlinear_trainreport)
#Train Accurcy is 98%

poly=SVC(kernel="poly")
poly.fit(X_train_s,Y_train)
pred_poly_train=poly.predict
pred_poly_test=poly.predict(X_test_s)
confusion_matrix(pred_poly_test,Y_test)

svmp_trainreport=classification_report(pred_poly_train(X_train_s),Y_train)
print(svmp_trainreport)
#Accuracy of train data is 98%

svmp_testreport=classification_report(pred_poly_test,Y_test)
print(svmp_testreport)
#Accuracy of test data is 98%

rbf=SVC(kernel="rbf")
rbf.fit(X_train_s,Y_train)
pred_rbf_train=rbf.predict(X_train_s)
pred_rbf_test=rbf.predict(X_test_s)
confusion_matrix(pred_rbf_test,Y_test)

cls_report_rbf_test=classification_report(Y_test,pred_rbf_test,output_dict=True)
print(cls_report_rbf_test)
#Test Accuracy is 98.48%

cls_report_rbf_train=classification_report(Y_train,pred_rbf_train)
print(cls_report_rbf_train)
#Accurcy of train data is  99%

#Neural Network
import tensorflow as tf
from keras.models import Sequential
conda install -c conda-forge tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
#from keras.layers import Dense,Activation,Layers,Lambda

model=Sequential()
model.add(layers.Dense(500,input_dim=14,activation='relu'))
model.add(layers.Dense(500,activation='sigmoid'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics='accuracy')
model.fit(X_train_s,Y_train,epochs=500,batch_size=1000)
  
accuracy_train=model.evaluate(X_train_s,Y_train)
# Train data Accuracy is 99.34%

accuracy_test=model.evaluate(X_test_s,Y_test)
#Test data Accuracy is 98.67%

predictions = model.predict_classes(X_train)
predictions
confusion_matrix(predictions,Y_train)


def NN_model(hidden_layer):
    model=Sequential()
    for i in range(1,len(hidden_layer)-1):
        if (i==1):
            model.add(layers.Dense(hidden_layer[i],input_dim=hidden_layer[0],activation='relu'))
        else:
            model.add(layers.Dense(hidden_layer[i]))
    model.add(layers.Dense(hidden_layer[-1],kernel_initializer="normal",activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics="accuracy")
    return model

model_1=NN_model([66,120,100,80,1])
model_1.fit(np.array(X_train),np.array(Y_train),epochs=500)
pred_train=model_1.predict(np.array(X_train))
predict_train=pd.Series(i[0] for i in pred_train)

# Bagging and Boosting

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=50)
bg.fit(X_train,Y_train)

bg.score(X_test,Y_test)         #Test Accuracy is 99%
bg.score(X_train,Y_train)       #Train Accuracy is 99.3%

ada=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=50,learning_rate=1.0)
ada.fit(X_train,Y_train)

ada.score(X_test,Y_test)        #Test Accuracy is 98.93%
ada.score(X_train,Y_train)      #Train Accuracy is 99.99%

#Voting
evc=VotingClassifier(estimators=[('Logreg',lr),('dt',dt),(rf,'rf'),('linear',svm)],voting='hard')

#Preparing XGB Classifier
pip install xgboost
import xgboost as xgb

xgboost=xgb.XGBClassifier(n_estimators=100,learning_rate=0.03,max_depth=4,max_leaves=10)
xgboost.fit(X_train,Y_train)
xgb_train_prd=xgboost.predict(X_train)
xgb_train_prd=pd.DataFrame(xgb_train_prd)

xgb_test_prd=xgboost.predict(X_test)
xgb_test_prd=pd.DataFrame(xgb_test_prd)

#train accuracy
train_accuracy=np.mean(xgb_train_prd.values==Y_train)
train_accuracy
# Train data Accuracy is 99%


#test accuracy
test_accuracy=np.mean(Y_test==xgb_test_prd.values)
test_accuracy
# Test data Accuracy is 99%

confusion_matrix(xgb_train_prd,Y_train)
