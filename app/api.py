
# import packages
from flask import Flask, jsonify, request
from storage import store_data, check_data, store_data_withfeatures
from model import model_predict

# paths
FOLDER_PATH = "/Users/ncv/github"
SQLITE_PATH = FOLDER_PATH + "/fraud-detection-api/data/fraud.sqlite"
MODEL_PATH = FOLDER_PATH + "/fraud-detection-api/data/fraud_classifier.pkl"

# initialize flask app
app = Flask(__name__)


# fraud endpoint
@app.route('/is-fraud', methods=['POST'])
def check_fraud():

    # get body posted
    q = request.json

    # check if columns are complete
    checked = check_data(q)
    if "ERROR" in checked:
        return jsonify(
            data={
                "error": checked
            })

    else:

        # store data
        store_data(SQLITE_PATH, q)

        # add features
        df_wfeatures = store_data_withfeatures(SQLITE_PATH, "features", q)

        # predict if fraud
        prediction = model_predict(MODEL_PATH, SQLITE_PATH, df_wfeatures)

        # return outcome
        return jsonify(
            data={
                "isFraud": "true" if prediction == 1 else 'false'})


if __name__ == '__main__':
    app.run(debug=True, port=2468)
