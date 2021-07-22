from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
model = pickle.load(open('fraudulent.pkl', 'rb'))
df= pd.read_csv('transaction.csv')
X=df.drop(['isFraud'], axis = 1)

y=df[['isFraud']]
# Convert boolean features into int variables

boo = X.select_dtypes(include=['bool']).columns.to_list()
for column in boo:
    X[column] = X[column].astype(int)
X.reversed = X.reversed.astype(int)
X.multi = X.multi.astype(int)
y.isFraud=y.isFraud.astype(int)

# Encoding the Features
encoder=ColumnTransformer([('encoder',OneHotEncoder(), [4,5])],remainder='passthrough')
X=encoder.fit_transform(X)


def index(request):
    return render(request,'index.html')

def prediction_model(l1):
    pred = np.array(l1)
    pred = pred.reshape(1,17)
    X_ui = encoder.transform(pred)

    temp1=X_ui[:, 1:19]
    temp2=X_ui[:,20:]

    # Concatenating temp1 and temp2
    X_ui = np.concatenate((temp1, temp2), axis=1)

    output = model.predict(X_ui)

    print("Response", output)
    return output




def result(request):
    creditLimit=int(request.POST.get('creditLimit'))
    transactionAmount=int(request.POST.get('transactionAmount'))
    posEntryMode=int(request.POST.get('posEntryMode'))
    posConditionCode=int(request.POST.get('posConditionCode'))
    merchantCategoryCode=request.POST.get('merchantCategoryCode')
    transactionType=request.POST.get('transactionType')
    currentBalance=int(request.POST.get('currentBalance'))
    cardPresent=int(request.POST.get('cardPresent'))
    expirationDateKeyInMatch=int(request.POST.get('expirationDateKeyInMatch'))
    rightCVV=int(request.POST.get('rightCVV'))
    sameCountry=int(request.POST.get('sameCountry'))
    firPurchase=int(request.POST.get('firPurchase'))
    expTime=int(request.POST.get('expTime'))
    openTime=int(request.POST.get('openTime'))
    changeAddTime=int(request.POST.get('changeAddTime'))
    reversed=int(request.POST.get('reversed'))
    multi=int(request.POST.get('multi'))

    l1=[creditLimit,transactionAmount,posEntryMode,posConditionCode,merchantCategoryCode,transactionType,currentBalance,cardPresent,
      expirationDateKeyInMatch,rightCVV,sameCountry,firPurchase,expTime,openTime,
      changeAddTime,reversed,multi]
    output=prediction_model(l1)
    print(output)
    if output == [1]:
        prediction = "The Given Transaction is Fraudulent."
    else:
        prediction = "The Given Transaction is not Fraudulent."

    print(prediction)
    passs = {'prediction_text': 'Model has Predicted', 'output': prediction}
    return render(request,'result.html',passs)