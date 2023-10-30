import pickle

from flask import Flask ,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


#import ridge regressor model and standard Scaler pickle

ridge_model=pickle.load(open('model/ridge.pkl','rb'))
standard_scaler=pickle.load(open('model/scaler.pkl','rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':

        #request.form.get('Temperature' -- will fetch the data which will be given by the user
        # important : Maintain the same Columns serials which are mentioned in the datasets 
        # Otherwise , Predicted data will  be not be predicted Correct 

        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    # result value will be in List So we have used [0] and there will be only 1 out in the result 
    else:
        return render_template('home.html')





if __name__=="__main__":
    app.run(host="0.0.0.0")
