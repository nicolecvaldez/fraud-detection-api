
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)

con = sqlite3.connect("/Users/nicole/github/fraud-detection-api/data/fraud.sqlite")
cur = con.cursor()

# ingested

cur.execute("CREATE TABLE ingested (key, step, type, amount, nameOrig, oldbalanceOrig, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest);")
df = pd.read_csv("/Users/nicole/github/fraud-detection-api/data/prod/ingested.csv")
df["key"] = df["index"]
df = df[['key', 'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrig', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']]
df.to_sql("ingested", con, if_exists='append', index=False)

df_check = pd.read_sql_query("select * from ingested", con)

# features

cur.execute("CREATE TABLE features (key, step, type, amount, nameOrig, oldbalanceOrig, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, type_IN, type_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, same_sender_perStep, same_sender_type_perStep, same_sender_recipient_perStep, same_sender_amount_perStep, same_sender_type_recipient_perStep, same_sender_type_amount_perStep, same_sender_recipient_amount_perStep, same_sender_type_recipient_amount_perStep, same_recipient_perStep, same_recipient_type_perStep, same_recipient_amount_perStep, same_recipient_type_amount_perStep, same_senderbal_txnamount, senderbal_greaterthan10M);")
df = pd.read_csv("/Users/nicole/github/fraud-detection-api/data/prod/features.csv")
df["key"] = df["index"]
df = df[[
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
df.to_sql("features", con, if_exists='append', index=False)

df_check = pd.read_sql_query("select * from features", con)

# predicted

cur.execute("CREATE TABLE predicted (key, isFraud);")
df = pd.read_csv("/Users/nicole/github/fraud-detection-api/data/prod/predicted.csv")
df["key"] = df["index"]
df = df[['key', 'isFraud']]
df.to_sql("predicted", con, if_exists='append', index=False)

df_check = pd.read_sql_query("select * from predicted", con)

# show tables

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

# drop other tables

cur.execute("DROP TABLE t")

# close connection

con.close()