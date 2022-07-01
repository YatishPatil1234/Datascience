import pandas as pd
df=pd.read_csv("C:/Users/91775/Downloads/Dataset/WineQT.csv")
print(df)


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



lr = LinearRegression()


x =df.drop('quality',axis=1)
y = df['quality'] #.apply(lambda y_value: 1 if y_value>=7 else 0)


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

lr = lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)

print("Accuracy score:",mean_squared_error(y_test,lr_pred))

"""
OUTPUT:
     fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality    Id
0               7.4             0.700         0.00             1.9      0.076                 11.0                  34.0  0.99780  3.51       0.56      9.4        5     0
1               7.8             0.880         0.00             2.6      0.098                 25.0                  67.0  0.99680  3.20       0.68      9.8        5     1
2               7.8             0.760         0.04             2.3      0.092                 15.0                  54.0  0.99700  3.26       0.65      9.8        5     2
3              11.2             0.280         0.56             1.9      0.075                 17.0                  60.0  0.99800  3.16       0.58      9.8        6     3
4               7.4             0.700         0.00             1.9      0.076                 11.0                  34.0  0.99780  3.51       0.56      9.4        5     4
...             ...               ...          ...             ...        ...                  ...                   ...      ...   ...        ...      ...      ...   ...
1138            6.3             0.510         0.13             2.3      0.076                 29.0                  40.0  0.99574  3.42       0.75     11.0        6  1592
1139            6.8             0.620         0.08             1.9      0.068                 28.0                  38.0  0.99651  3.42       0.82      9.5        6  1593
1140            6.2             0.600         0.08             2.0      0.090                 32.0                  44.0  0.99490  3.45       0.58     10.5        5  1594
1141            5.9             0.550         0.10             2.2      0.062                 39.0                  51.0  0.99512  3.52       0.76     11.2        6  1595
1142            5.9             0.645         0.12             2.0      0.075                 32.0                  44.0  0.99547  3.57       0.71     10.2        5  1597

[1143 rows x 13 columns]
Accuracy score: 0.3609981256364635
"""


