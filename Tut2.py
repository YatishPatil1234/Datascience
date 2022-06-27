import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn .ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn .ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import  MultinomialNB

df=pd.read_csv("C:/Users/CC-112/Desktop/IRIS.csv")

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nb=MultinomialNB()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)


x=df.drop("species",axis=1)
y=df["species"]
'''print(x)
print(y)'''

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

rf.fit(x_train,y_train)
r_pred=rf.predict(x_test)

dt.fit(x_train,y_train)
d_pred=dt.predict(x_test)

nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)

nn.fit(x_train,y_train)
nn_pred=nn.predict(x_test)

gb.fit(x_train,y_train)
gb_pred=gb.predict(x_test)

sv.fit(x_train,y_train)
sv_pred=sv.predict(x_test)


print("Logistic Regression:",accuracy_score(y_test,y_pred))
print("Random Forest:",accuracy_score(y_test,r_pred))
print("Decision Tree:",accuracy_score(y_test,d_pred))
print("Naive Baye's:",accuracy_score(y_test,nb_pred))
print("Neural Network:",accuracy_score(y_test,nn_pred))
print("Gradient Forest :",accuracy_score(y_test,gb_pred))
print("SVM:",accuracy_score(y_test,sv_pred))

'''
OUTPUT:
Logistic Regression: 0.9777777777777777
Random Forest: 0.9777777777777777
Decision Tree: 0.9777777777777777
Naive Baye's: 0.6
Neural Network: 0.24444444444444444
Gradient Forest : 0.9777777777777777
SVM: 0.9777777777777777
'''



