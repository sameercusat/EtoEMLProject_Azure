from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import DataReading,DataPrediction
from src.exception import CustomException
import sys

application = Flask(__name__)
app=application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictscore',methods=['GET','POST'])
def predict_math_score():
    if request.method == 'GET':
        return render_template('predict_score.html',result=0)
    else:
            try:    
                gender = request.form.get('gender')
                race = request.form.get('race')
                education = request.form.get('education')
                lunch = request.form.get('lunch')
                course = request.form.get('course')
                rscore = request.form.get('rscore')
                wscore = request.form.get('wscore')
                data_reader = DataReading(gender,race,education,lunch,course,rscore,wscore)
                df = data_reader.create_dataframe()
                data_predictor = DataPrediction()
                result = data_predictor.initiate_data_prediction(df)
                return render_template('predict_score.html',result=result)
            except Exception as e:
                 raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(debug=True)