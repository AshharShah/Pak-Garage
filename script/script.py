import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset/used_car_data_cleaned.csv')

df = df.iloc[:, 1:]

# now split the dataframe into variables that are used to develop the model
x = df.iloc[:, 1:]
y = df.loc[:, ['Price']]
# use the split function to break the data further
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .75)
# use the decision tree regression model
regr = DecisionTreeRegressor(max_depth=8)
regr.fit(X_train, y_train)
# use our created model to make predictions on the test set
y_pred = regr.predict(X_test)
# convert the test output into 1D numpy array for finding score
y_testnp = y_test.to_numpy().flatten()
# calculating the score / how good our model performed
# compare the actual values with the predicted values
score = r2_score(y_testnp, y_pred)
print('Model Accuracy Score: ' + str(score))

mname = 'Honda'
myear = 2012
mil = 80000
eng_cap = 1300
city = 'Lahore'
etype = 'Petrol'
btyp = 'Sedan'
trans = 'Manual'

topred = np.array([myear,mil,eng_cap])

if (city == 'Lahore'):
    merg = np.array([[1,0,0,0]])
    topred = np.concatenate((topred, merg), axis=None)
elif (city == 'Islamabad'):
    merg = np.array([[0,1,0,0]])
    topred = np.concatenate((topred, merg), axis=None)
elif (city == 'Karachi'):
    merg = np.array([[0,0,1,0]])
    topred = np.concatenate((topred, merg), axis=None)
elif (city == 'Un-Registered'):
    merg = np.array([0,0,0,1])
    topred = np.concatenate((topred, merg), axis=None)
else:
    merg = np.array([0,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)

print(topred)

if (etype == 'Petrol'):
    merg = np.array([1,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (etype == 'Hybrid'):
    merg = np.array([0,1,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (etype == 'Diesel'):
    merg = np.array([0,0,1])
    topred = np.concatenate((topred, merg), axis=None)
else:
    merg = np.array([0,0,0])
    topred = np.concatenate((topred, merg), axis=None)

if (trans == 'Manual'):
    merg = np.array([1,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (trans == 'Automatic'):
    merg = np.array([0,1])
    topred = np.concatenate((topred, merg), axis=None)
else:
    merg = np.array([0,0])
    topred = np.concatenate((topred, merg), axis=None)

print(topred)

if (btyp == 'Hatchback'):
    merg = np.array([1,0,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (btyp == 'Sedan'):
    merg = np.array([0,1,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (btyp == 'SUV'):
    merg = np.array([0,0,1,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (btyp == 'Mini Van'):
    merg = np.array([0,0,0,1,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (btyp == 'Crossover'):
    merg = np.array([0,0,0,0,1])
    topred = np.concatenate((topred, merg), axis=None)
else:
    merg = np.array([0,0,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)

if (mname == 'Suzuki'):
    merg = np.array([1,0,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (mname == 'Toyota'):
    merg = np.array([0,1,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (mname == 'Honda'):
    merg = np.array([0,0,1,0,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (mname == 'Daihatsu'):
    merg = np.array([0,0,0,1,0])
    topred = np.concatenate((topred, merg), axis=None)
elif (mname == 'Nissan'):
    merg = np.array([0,0,0,0,1])
    topred = np.concatenate((topred, merg), axis=None)
else:
    merg = np.array([0,0,0,0,0])
    topred = np.concatenate((topred, merg), axis=None)

print(topred)

#columns = np.array(['Price', 'Model Year', 'Mileage', 'Eng_Cap', 'City_Lahore', 'City_Islamabad', 'City_Karachi', 'City_Un-Registered', 'Eng_Type_Petrol', 'Eng_Type_Hybrid', 'Eng_Type_Diesel', 'Transmission_Manual', 'Transmission_Automatic', 'Body_Type_Hatchback', 'Body_Type_Sedan', 'Body_Type_SUV', 'Body_Type_Mini Van', 'Body_Type_Crossover', 'Make_Suzuki', 'Make_Toyota', 'Make_Honda', 'Make_Daihatsu', 'Make_Nissan'])

B = np.reshape(topred, (-1, 22))

print(B)

ndf = pd.DataFrame(B, columns = ['Model Year', 'Mileage', 'Eng_Cap', 'City_Lahore', 'City_Islamabad', 'City_Karachi', 'City_Un-Registered', 'Eng_Type_Petrol', 'Eng_Type_Hybrid', 'Eng_Type_Diesel', 'Transmission_Manual', 'Transmission_Automatic', 'Body_Type_Hatchback', 'Body_Type_Sedan', 'Body_Type_SUV', 'Body_Type_Mini Van', 'Body_Type_Crossover', 'Make_Suzuki', 'Make_Toyota', 'Make_Honda', 'Make_Daihatsu', 'Make_Nissan'])

topred = ndf.iloc[:, :]

prediction = regr.predict(topred)
print('Predicted Price: ' + str(prediction[0]))