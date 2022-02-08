
# import packages
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)


DATA_COLS = {
    'step': 'int',
    'type': 'str',
    'amount': 'float',
    'nameOrig': 'str',
    'oldbalanceOrig': 'float',
    'newbalanceOrig': 'float',
    'nameDest': 'str',
    'oldbalanceDest': 'float',
    'newbalanceDest': 'float',
                }


def check_data(dict_append):
    """

    :param dict_append:
    :return:
    """

    if sorted(list(dict_append.keys())) != sorted(list(DATA_COLS.keys())):
        return "ERROR: Incorrect data"
    else:
        return "SUCCESS: Correct data"


def store_data(data_path, dict_append):
    """

    :param data_path:
    :param dict_append:
    :return:
    """

    if ".csv" in data_path:

        # prepare new key and add to dictionary
        max_index_ = pd.read_csv(data_path)
        max_index = max_index_["key"].max()
        dict_append["key"] = max_index.loc[0][0] + 1

        # convert dictionary to dataframe
        df_append = pd.DataFrame(dict_append, index=[0])
        df_append = df_append[[
            'key', 'step', 'type', 'amount', 'nameOrig',
            'oldbalanceOrig', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
        ]]

        # add new dataset
        df_append.to_csv(data_path, mode='a', index=False, header=False)

    elif ".sqlite" in data_path:

        # connect to dataset
        con = sqlite3.connect(data_path)

        # prepare new key and add to dictionary
        max_index = pd.read_sql_query("select max(key) as max_key from ingested", con)
        dict_append["key"] = max_index.loc[0][0]+1

        # convert dictionary to dataframe
        df_append = pd.DataFrame(dict_append, index=[0])
        df_append = df_append[[
            'key', 'step', 'type', 'amount', 'nameOrig',
            'oldbalanceOrig', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
        ]]

        # add new dataset
        df_append.to_sql("ingested", con, if_exists='append', index=False)

        # disconnect to dataset
        con.close()

    else:

        # better storage types - psql
        pass

    return "Data stored."


def store_data_withfeatures(data_path, features_table, dict_append):
    """

    :param data_path:
    :param features_table:
    :param dict_append:
    :return:
    """

    # connect to dataset
    con = sqlite3.connect(data_path)

    # list features to add
    feature_list = [
        "type_IN", "type_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
        'same_sender_perStep', 'same_sender_type_perStep',
        'same_sender_recipient_perStep', 'same_sender_amount_perStep',
        'same_sender_type_recipient_perStep', 'same_sender_type_amount_perStep',
        'same_sender_recipient_amount_perStep', 'same_sender_type_recipient_amount_perStep',
        'same_recipient_perStep', 'same_recipient_type_perStep',
        'same_recipient_amount_perStep', 'same_recipient_type_amount_perStep',
        "same_senderbal_txnamount", "senderbal_greaterthan10M"]
    for f_l in feature_list:
        dict_append[f_l] = 0

    # type
    if dict_append["type"] == "CASH_IN":
        dict_append["type_IN"] += 1
    elif dict_append["type"] == "CASH_OUT":
        dict_append["type_OUT"] += 1
    elif dict_append["type"] == "DEBIT":
        dict_append["type_DEBIT"] += 1
    elif dict_append["type"] == "PAYMENT":
        dict_append["type_PAYMENT"] += 1
    elif dict_append["type"] == "TRANSFER":
        dict_append["type_TRANSFER"] += 1
    else:
        pass

    # transfer amount = balance
    if dict_append["oldbalanceOrig"] == dict_append["amount"]:
        dict_append["same_senderbal_txnamount"] += 1
    else:
        pass

    # balance > 10M
    if dict_append["oldbalanceOrig"] > 10000000:
        dict_append["senderbal_greaterthan10M"] += 1
    else:
        pass

    # txn sequence

    # 1
    same_sender_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s"
        % (features_table, dict_append['nameOrig'], dict_append['step']), con)
    if not same_sender_step.empty:
        dict_append["same_sender_perStep"] = same_sender_step["same_sender_perStep"].max()+1
    else:
        dict_append["same_sender_perStep"] = 1

    # 2
    same_sender_type_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and type='%s'"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["type"]), con)
    if not same_sender_type_step.empty:
        dict_append["same_sender_type_perStep"] = same_sender_type_step["same_sender_type_perStep"].max()+1
    else:
        dict_append["same_sender_type_perStep"] = 1

    # 3
    same_sender_recipient_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and nameDest='%s'"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["nameDest"]), con)
    if not same_sender_recipient_step.empty:
        dict_append["same_sender_recipient_perStep"] = same_sender_recipient_step["same_sender_recipient_perStep"].max()+1
    else:
        dict_append["same_sender_recipient_perStep"] = 1

    # 4
    same_sender_amount_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and amount=%s"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["amount"]), con)
    if not same_sender_amount_step.empty:
        dict_append["same_sender_amount_perStep"] = same_sender_amount_step["same_sender_amount_perStep"].max()+1
    else:
        dict_append["same_sender_amount_perStep"] = 1

    # 5
    same_sender_type_recipient_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and type='%s' and nameDest='%s'"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["type"], dict_append["nameDest"]), con)
    if not same_sender_type_recipient_step.empty:
        dict_append["same_sender_type_recipient_perStep"] = same_sender_type_recipient_step["same_sender_type_recipient_perStep"].max()+1
    else:
        dict_append["same_sender_type_recipient_perStep"] = 1

    # 6
    same_sender_type_amount_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and type='%s' and amount=%s"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["type"], dict_append["amount"]), con)
    if not same_sender_type_amount_step.empty:
        dict_append["same_sender_type_amount_perStep"] = same_sender_type_amount_step["same_sender_type_amount_perStep"].max()+1
    else:
        dict_append["same_sender_type_amount_perStep"] = 1

    # 7
    same_sender_recipient_amount_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and type='%s' and nameDest='%s'"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["type"], dict_append["nameDest"]), con)
    if not same_sender_recipient_amount_step.empty:
        dict_append["same_sender_recipient_amount_perStep"] = same_sender_recipient_amount_step["same_sender_recipient_amount_perStep"].max()+1
    else:
        dict_append["same_sender_recipient_amount_perStep"] = 1

    # 8
    same_sender_type_recipient_amount_step = pd.read_sql_query(
        "select * from %s where nameOrig='%s' and step=%s and type='%s' and nameDest='%s' and amount=%s"
        % (features_table, dict_append['nameOrig'], dict_append['step'], dict_append["type"], dict_append["nameDest"], dict_append["amount"]), con)
    if not same_sender_type_recipient_amount_step.empty:
        dict_append["same_sender_type_recipient_amount_perStep"] = same_sender_type_recipient_amount_step["same_sender_type_recipient_amount_perStep"].max()+1
    else:
        dict_append["same_sender_type_recipient_amount_perStep"] = 1

    # 9
    same_recipient_step = pd.read_sql_query(
        "select * from %s where nameDest='%s' and step=%s"
        % (features_table, dict_append['nameDest'], dict_append['step']), con)
    if not same_recipient_step.empty:
        dict_append["same_recipient_perStep"] = same_recipient_step["same_recipient_perStep"]+1
    else:
        dict_append["same_recipient_perStep"] = 1

    # 10
    same_recipient_type_step = pd.read_sql_query(
        "select * from %s where nameDest='%s' and step=%s and type='%s'"
        % (features_table, dict_append['nameDest'], dict_append['step'], dict_append["type"]), con)
    if not same_recipient_type_step.empty:
        dict_append["same_recipient_type_perStep"] = same_recipient_type_step["same_recipient_type_perStep"].max()+1
    else:
        dict_append["same_recipient_type_perStep"] = 1

    # 11
    same_recipient_amount_step = pd.read_sql_query(
        "select * from %s where nameDest='%s' and step=%s and amount=%s"
        % (features_table, dict_append['nameDest'], dict_append['step'], dict_append["amount"]), con)
    if not same_recipient_amount_step.empty:
        dict_append["same_recipient_amount_perStep"] = same_recipient_amount_step["same_recipient_amount_perStep"].max()+1
    else:
        dict_append["same_recipient_amount_perStep"] = 1

    # 12
    same_recipient_type_amount_step = pd.read_sql_query(
        "select * from %s where nameDest='%s' and step=%s and amount=%s and type='%s'"
        % (features_table, dict_append['nameDest'], dict_append['step'], dict_append["amount"], dict_append["type"]), con)
    if not same_recipient_type_amount_step.empty:
        dict_append["same_recipient_type_amount_perStep"] = same_recipient_type_amount_step["same_recipient_type_amount_perStep"].max()+1
    else:
        dict_append["same_recipient_type_amount_perStep"] = 1

    max_index = pd.read_sql_query("select max(key) as max_key from ingested", con)

    dict_append["key"] = max_index.loc[0][0] + 1

    df_append = pd.DataFrame(dict_append, index=[0])

    df_append = df_append[[
        'key', 'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrig',
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
        'type_IN', 'type_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
        'same_sender_perStep', 'same_sender_type_perStep',
        'same_sender_recipient_perStep', 'same_sender_amount_perStep',
        'same_sender_type_recipient_perStep', 'same_sender_type_amount_perStep',
        'same_sender_recipient_amount_perStep',
        'same_sender_type_recipient_amount_perStep', 'same_recipient_perStep',
        'same_recipient_type_perStep', 'same_recipient_amount_perStep',
        'same_recipient_type_amount_perStep', 'same_senderbal_txnamount',
        'senderbal_greaterthan10M']]

    # add new dataset
    df_append.to_sql(features_table, con, if_exists='append', index=False)

    # disconnect to dataset
    con.close()

    return df_append
