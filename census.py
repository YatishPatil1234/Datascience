import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/91775/Downloads/Dataset/adult.csv")
reg = LinearRegression()
print(df.info())


le = preprocessing.LabelEncoder()
df['workclass']= le.fit_transform(df['workclass'])
df['education']= le.fit_transform(df['education'])
df['marital.status']= le.fit_transform(df['marital.status'])
df['occupation']= le.fit_transform(df['occupation'])
df['relationship']= le.fit_transform(df['relationship'])
df['race']= le.fit_transform(df['race'])
df['sex']= le.fit_transform(df['sex'])
df['native.country']= le.fit_transform(df['native.country'])
df['income']= le.fit_transform(df['income'])


x = df.drop('income',axis=1)
y = df['income']


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print('mean square error: ',mean_squared_error(y_test,reg_pred))

"""
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   age             32561 non-null  int64
 1   workclass       32561 non-null  object
 2   fnlwgt          32561 non-null  int64
 3   education       32561 non-null  object
 14  income          32561 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
None
mean square error:  0.13523685043672698

"""