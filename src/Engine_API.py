import joblib
import pandas as pd
from src.MLpipeline.Fraud import Fraud
from flask import Flask, request, Response

model = joblib.load('../output/model_cycle1.joblib')

app = Flask(__name__)

@app.route('/fraud/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = Fraud()

        df1 = pipeline.data_cleaning(test_raw)

        df2 = pipeline.feature_engineering(df1)

        df3 = pipeline.data_preparation(df2)

        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')
@app.route('/', methods=['POST'])
def index():
    return {"msg" : "good request"}

if __name__ == '__main__':
    app.run('0.0.0.0')