from flask import Flask , request,jsonify

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import requests
import csv

from openpyxl import Workbook
# import xlsxwriter


app = Flask(__name__)

@app.route('/getpredicteddata' , methods=['POST'])
def hello():

 data = request.get_json()

 headers = data

 listnew =[""]
 

 for x in data:
    
   listnew.append(x)

 expensetypes=[
    ["FOOD"],
    ["ENTERTAINMENT"],
    ["GROCERIES"],
   [ "EXTRA"],
    ["CAR"],
    ["LIVING"],
    ["GYM"],
    ["MEDICAL"],
    ["CLOTHING"],
    ["EDUCATION"]
 ]
 

 with open("expense.csv" , "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(listnew)
    writer.writerows(expensetypes)

 with open("expense.csv" , "r+" , newline="")as file:
    writer = csv.writer(file)
    reader = csv.reader(file)

    headers = next(reader)
    datalop = list(reader)

    for x in headers[1:]:
       for index,j in enumerate(datalop):
         #  print(index,j)
          for date,categories in data.items():
             if(date == x): 
                for types,values in categories.items():
                    
                    if(types==j[0]):
                     df = pd.read_csv('expense.csv')
                     df.at[index,x] = values
                     df.to_csv("expense.csv" ,index=False)

   
    for x in headers[1:]:
       for index,j in enumerate(datalop):
          df = pd.read_csv('expense.csv')
          if(pd.isna(df.at[index,x])):
             df.at[index,x] = 0
             df.to_csv("expense.csv" , index=False)


    dataset = pd.read_csv('expense.csv')
 
    num = dataset.shape[1]
 
    X = dataset.drop([dataset.columns[0], dataset.columns[num-1]],axis=1) 
    Y = dataset[dataset.columns[num-1]]

    names = dataset[dataset.columns[0]]

    X_train ,X_test,Y_train,Y_test = train_test_split(X,Y , test_size=0.1 , random_state=2)

    ling_reg_model = LinearRegression()

    ling_reg_model.fit(X_train , Y_train)
    predictedvalues=[]
    for index,row in X.iterrows():
       df = pd.DataFrame([row])
       training_data_prediction = ling_reg_model.predict(df)
       predictedvalues.append({"type":names[index] , "prediction":training_data_prediction[0]})

      
 return predictedvalues


@app.route('/hellow' , methods=['GET'])
def hello():
   return jsonify({"message":"hi"})


if __name__ == '__main__':
    app.run(debug=True)