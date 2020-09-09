#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
data_set="http://bit.ly/w-data"
data=pd.read_csv(data_set)



#ploting dataset
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


#Preparing the data
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


#Splitting of dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("training complete")


#Plotting the regression line
line=regressor.coef_*X_train+regressor.intercept_
plt.scatter(X_train,y_train)
plt.plot(X_train,line)
plt.show()



#Making predictions
y_pred = regressor.predict(X_test)



#Comparing actual vs predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)



#Testing algorithm with new data
hours=9.8
new_pred=regressor.predict([[hours]])
print(f"The predicted score for studying {hours}hr is: {new_pred}")



#Performance evaluation of the algorithm
from sklearn import metrics
print("Performance Evaluation")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))