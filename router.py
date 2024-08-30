import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
model = joblib.load('random_forest.pkl')
from utils import preprocess_new


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        satis = float(request.form['satis'])
        eval = float(request.form['eval'])
        num_pro = int(request.form['num_pro'])
        avg_mon_h = float(request.form['avg_mon_h'])
        t_spend_comp = int(request.form['t_spend_comp'])
        promotion_last_5years = request.form['promotion_last_5years']
        Work_acc = request.form['Work_acc']
        salary = request.form['salary']









        # Remmber the Feature Engineering we did
        projects_years = t_spend_comp / num_pro
        month_project = num_pro / avg_mon_h
    

        # Concatenate all Inputs
        X_new = pd.DataFrame({'satisfaction_level': [satis], 'last_evaluation': [eval], 'number_project': [num_pro], 'average_montly_hours': [avg_mon_h],
                              'time_spend_company': [t_spend_comp], 'Work_accident': [Work_acc], 'promotion_last_5years': [promotion_last_5years], 'projects_years': [projects_years],
                              'month_project': [month_project],'salary':[salary]
                              })
        
        X_processed = preprocess_new(X_new)



   
        

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])
    

        return render_template('predict.html', pred_vals=y_pred_new)
    else:
        return render_template('predict.html')


@app.route('/about')
def about():
    return render_template('about.html')






if __name__ =='__main__':
    app.run(debug = True)