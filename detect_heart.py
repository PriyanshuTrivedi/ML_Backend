import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler


model=pickle.load(open("test.pkl","rb"))

def detectHeart(data):
    values={
        'age':eval(data['age']),
        'sex':eval(data['sex']),
        'trestbps':eval(data['trestbps']),
        'chol':eval(data['chol']),
        'fbs':eval(data['fbs']),
        'restecg':eval(data['restecg']),
        'thalach':eval(data['thalach']),
        'exang':eval(data['exang']),
        'oldpeak':eval(data['oldpeak']),
        'ca':eval(data['ca']),
        'cp_0':0,
        'cp_1':0,
        'cp_2':0,
        'cp_3':0,
        'thal_0':0,
        'thal_1':0,
        'thal_2':0,
        'thal_3':0,
        'slope_0':0,
        'slope_1':0,
        'slope_2':0,
    }

    values[f'cp_{data["cp"]}']=values[f'thal_{data["thal"]}']=values[f'slope_{data["slope"]}']=1


    output=model.predict([np.array(list(values.values()))])
    
    return output

        
