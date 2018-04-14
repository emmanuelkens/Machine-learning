#import sklearn, numpy and pandas libraries
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
#define rows using numpy
row1 = np.array([2000,20,100])
row2 = np.array([2001,25,150])
row3 = np.array([2002,30,200])
row4 = np.array([2003,35,250])
data = np.array([row1,row2,row3,row4])
columns = np.array(['year','workers','results'])
#define dataframe using pandas
df = pd.DataFrame(data,columns=columns)
#print data frame
print(df)
#drop results column because we are going to predict it
X_train = df.drop(columns=['results'])
print(X_train)
#drop other columns so that we can remain with y_train column which is a dependent variable
y_train = df.drop(columns=['year','workers'])
#plot
plt.plot(df['year'],df['results'])
plt.ylabel("Tonnes")
plt.xlabel("Year")
plt.show()
#create a linear model using linearRegression()
#fit our training data
clf = linear_model.LinearRegression().fit(X_train,y_train)
#use clf model to predict our results when we have X_test
#print our results
X_test = np.array([[2004,40]])
print(clf.predict(X_test))
#check our accuracy using clf.score()
print(clf.score(X_train,y_train))
