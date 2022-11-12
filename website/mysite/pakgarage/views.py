from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.template import loader

# libraries for our ML model and processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

@csrf_exempt
def index(request):
    return render(request,'index.html')
@csrf_exempt
def about(request):
    return render(request,'about.html')

@csrf_exempt
def output(request):
    print("\n")
    print("Recieved Data!")
    print("Manufacturer: " + str(request.POST.get("make", "")))
    print("Model Year: " + str(request.POST.get("modelYear", "")))
    print("Mileag: " + str(request.POST.get("mileage", "")))
    print("Engine Capacity: " + str(request.POST.get("engineCap", "")))
    print("Registration City: " + str(request.POST.get("regCity", "")))
    print("Engine Type: " + str(request.POST.get("engType", "")))
    print("Body Type: " + str(request.POST.get("bodyType", "")))
    print("Transmission: " + str(request.POST.get("transmission", "")))
    print("\n")
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
    print("\n")

    mname = str(request.POST.get("make", ""))
    myear = int(request.POST.get("modelYear", ""))
    mil = int(request.POST.get("mileage", ""))
    eng_cap = int(request.POST.get("engineCap", ""))
    city = str(request.POST.get("regCity", ""))
    etype = str(request.POST.get("engType", ""))
    btyp = str(request.POST.get("bodyType", ""))
    trans = str(request.POST.get("transmission", ""))
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


    #columns = np.array(['Price', 'Model Year', 'Mileage', 'Eng_Cap', 'City_Lahore', 'City_Islamabad', 'City_Karachi', 'City_Un-Registered', 'Eng_Type_Petrol', 'Eng_Type_Hybrid', 'Eng_Type_Diesel', 'Transmission_Manual', 'Transmission_Automatic', 'Body_Type_Hatchback', 'Body_Type_Sedan', 'Body_Type_SUV', 'Body_Type_Mini Van', 'Body_Type_Crossover', 'Make_Suzuki', 'Make_Toyota', 'Make_Honda', 'Make_Daihatsu', 'Make_Nissan'])

    B = np.reshape(topred, (-1, 22))


    ndf = pd.DataFrame(B, columns = ['Model Year', 'Mileage', 'Eng_Cap', 'City_Lahore', 'City_Islamabad', 'City_Karachi', 'City_Un-Registered', 'Eng_Type_Petrol', 'Eng_Type_Hybrid', 'Eng_Type_Diesel', 'Transmission_Manual', 'Transmission_Automatic', 'Body_Type_Hatchback', 'Body_Type_Sedan', 'Body_Type_SUV', 'Body_Type_Mini Van', 'Body_Type_Crossover', 'Make_Suzuki', 'Make_Toyota', 'Make_Honda', 'Make_Daihatsu', 'Make_Nissan'])

    topred = ndf.iloc[:, :]

    prediction = regr.predict(topred)
    prediction =int(prediction[0])
    prediction = f'{prediction:,}'
    print('Predicted Price: ' + str(prediction))

    context = {
        'man': mname,
        'myear': myear,
        'mil': mil,
        'ec' :eng_cap,
        'rc':city,
        'et':etype,
        'bt':btyp,
        'trans':trans,
        'exp': prediction
    }

    return render(request, 'output.html', context)