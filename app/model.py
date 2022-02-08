
# import packages
from sklearn.tree import DecisionTreeClassifier
import pickle
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)

#paths
FOLDER_PATH = "/Users/ncv/github"


def model_train(data_path):
    """

    :param data_path:
    :return:
    """

    # read training data csv
    train_data = pd.read_csv(data_path)

    features = [c for c in train_data.columns if c not in ["key", "nameOrig", "nameDest", "type", "isFraud"]]

    x_train = train_data[features].to_numpy()
    y_train = train_data[["isFraud"]].to_numpy()

    model = DecisionTreeClassifier(
        max_depth=10, random_state=101, max_features=None, min_samples_leaf=15
    )
    model.fit(x_train, y_train)

    with open('FOLDER_PATH'+'/fraud-detection-api/data/fraud_classifier.pkl', 'wb') as fid:
        pickle.dump(model, fid)

    return "Model trained and saved."


def model_predict(model_path, data_path, dataframe_topredict):
    """

    :param model_path:
    :param data_path:
    :param dataframe_topredict:
    :return:
    """

    # get stored model
    with open(model_path, 'rb') as fid:
        model = pickle.load(fid)

    # keep only columns with numerical values
    features = [c for c in dataframe_topredict.columns if c not in ["key", "nameOrig", "nameDest", "type"]]
    df_predict = dataframe_topredict[features]

    # predict if fraud
    pred = model.predict(df_predict.to_numpy())[0]

    # tag as fraud if multiple transactiosn with same step, sender, recipient, type and amount
    if dataframe_topredict["same_sender_type_recipient_amount_perStep"][0] != 1:
        pred = 1

    # store prediction
    con = sqlite3.connect(data_path)
    df_append = pd.DataFrame({
        "key": dataframe_topredict["key"],
        "isFraud": pred,
    }, index=[0])
    df_append.to_sql("predicted", con, if_exists='append', index=False)
    con.close()

    return pred
