
import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv("data/train/transactions_train.csv")

df["money_out"] = df["type"].apply(lambda x: 0 if x=="CASH_IN" else 1)

df.reset_index(inplace=True)

df["same_nameOrig_perStep"] = df.groupby(["step", "nameOrig"])["index"].rank("dense", ascending=False)
df["same_nameOrig_type_perStep"] = df.groupby(["step", "nameOrig", "type"])["index"].rank("dense", ascending=False)
df["same_nameOrig_nameDest_perStep"] = df.groupby(["step", "nameOrig", "nameDest"])["index"].rank("dense", ascending=False)
df["same_nameOrig_amount_perStep"] = df.groupby(["step", "nameOrig", "amount"])["index"].rank("dense", ascending=False)
df["same_nameOrig_type_nameDest_perStep"] = df.groupby(["step", "nameOrig", "type", "nameDest"])["index"].rank("dense", ascending=False)
df["same_nameOrig_type_amount_perStep"] = df.groupby(["step", "nameOrig", "type", "amount"])["index"].rank("dense", ascending=False)
df["same_nameOrig_nameDest_amount_perStep"] = df.groupby(["step", "nameOrig", "nameDest", "amount"])["index"].rank("dense", ascending=False)
df["same_nameOrig_type_nameDest_amount_perStep"] = df.groupby(["step", "nameOrig", "type", "nameDest", "amount"])["index"].rank("dense", ascending=False)
df["same_nameDest_perStep"] = df.groupby(["step", "nameDest"])["index"].rank("dense", ascending=False)

df.to_csv("data/train/transactions_train_addedfeatures.csv", index=False)

nonfraud = df[df["isFraud"]==0]
fraud = df[df["isFraud"]==1]

count_nameOrig = (df.groupby(["nameOrig"])["step"].count()).reset_index(name='count_transactions')
count_nameOrig_fraud = (fraud.groupby(["nameOrig"])["step"].count()).reset_index(name='count_transactions_fraud')
count_nameOrig_nonfraud = (nonfraud.groupby(["nameOrig"])["step"].count()).reset_index(name='count_transactions_nonfraud')
count_nameOrig_ = pd.merge(count_nameOrig, count_nameOrig_fraud, "left")
count_nameOrig__ = pd.merge(count_nameOrig_, count_nameOrig_nonfraud, "left")
count_nameOrig__.fillna(0, inplace=True)
count_nameOrig__.to_csv("data/train/nameOrig_count_transaction.csv", index=False)

count_nameDest = (df.groupby(["nameDest"])["step"].count()).reset_index(name='count_transactions')
count_nameDest_fraud = (fraud.groupby(["nameDest"])["step"].count()).reset_index(name='count_transactions_fraud')
count_nameDest_nonfraud = (nonfraud.groupby(["nameDest"])["step"].count()).reset_index(name='count_transactions_nonfraud')
count_nameDest_ = pd.merge(count_nameDest, count_nameDest_fraud, "left")
count_nameDest__ = pd.merge(count_nameDest_, count_nameDest_nonfraud, "left")
count_nameDest__.fillna(0, inplace=True)
count_nameDest__.to_csv("data/train/nameDest_count_transaction.csv", index=False)

fraud_sample = df[df["nameDest"]=='C803116137']

pre = df[df["step"]<320]
post = df[df["step"]>=320]
