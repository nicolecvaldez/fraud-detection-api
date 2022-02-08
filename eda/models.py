
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
import pandas as pd
pd.set_option('display.max_columns', None)

df_ = pd.read_csv("data/train/transactions_train.csv")

# type hot encode

encoder = OneHotEncoder()
# data_onehotencoded_list = [train]
data_onehotencoded_list = [df_]
for column in ["type"]:
    df_encoded_array = encoder.fit_transform(df_[[column]].to_numpy()).toarray()
    df_encoded = pd.DataFrame(df_encoded_array, columns = [column+"_"+x.split("_")[-1] for x in encoder.get_feature_names()])
    data_onehotencoded_list.append(df_encoded)
df = reduce(lambda left, right: pd.merge(left,right,left_index=True, right_index=True, how='outer'), data_onehotencoded_list)

# txn sequence

df.reset_index(inplace=True)

df["same_sender_perStep"] = df.groupby(["step", "nameOrig"])["index"].rank("dense", ascending=False)
df["same_sender_type_perStep"] = df.groupby(["step", "nameOrig", "type"])["index"].rank("dense", ascending=False)
df["same_sender_recipient_perStep"] = df.groupby(["step", "nameOrig", "nameDest"])["index"].rank("dense", ascending=False)
df["same_sender_amount_perStep"] = df.groupby(["step", "nameOrig", "amount"])["index"].rank("dense", ascending=False)
df["same_sender_type_recipient_perStep"] = df.groupby(["step", "nameOrig", "type", "nameDest"])["index"].rank("dense", ascending=False)
df["same_sender_type_amount_perStep"] = df.groupby(["step", "nameOrig", "type", "amount"])["index"].rank("dense", ascending=False)
df["same_sender_recipient_amount_perStep"] = df.groupby(["step", "nameOrig", "nameDest", "amount"])["index"].rank("dense", ascending=False)

df["same_sender_type_recipient_amount_perStep"] = df.groupby(["step", "nameOrig", "type", "nameDest", "amount"])["index"].rank("dense", ascending=False)

df["same_recipient_perStep"] = df.groupby(["step", "nameDest"])["index"].rank("dense", ascending=False)
df["same_recipient_type_perStep"] = df.groupby(["step", "nameDest", "type"])["index"].rank("dense", ascending=False)
df["same_recipient_amount_perStep"] = df.groupby(["step", "nameDest", "amount"])["index"].rank("dense", ascending=False)
df["same_recipient_type_amount_perStep"] = df.groupby(["step", "nameDest", "type", "amount"])["index"].rank("dense", ascending=False)

# transfer amount = balance

df["same_senderbal_txnamount"] = df.apply(lambda x: 1 if x["oldbalanceOrig"]==x["amount"] else 0, axis=1)

# balance > 10M

df["senderbal_greaterthan10M"] = df.apply(lambda x: 1 if x["oldbalanceOrig"]>10000000 else 0, axis=1)


# pre and post

post_df = df[df["step"]>=320]
pre_df = df[df["step"]<320]

# trim dataframe

X_train = pre_df[[
    'amount',
    # 'nameOrig',
    'oldbalanceOrig',
    'newbalanceOrig',
    # 'nameDest',
    'oldbalanceDest',
    'newbalanceDest',
    'type_IN',
    'type_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
    'same_sender_perStep',
    'same_sender_type_perStep',
    'same_sender_recipient_perStep',
    'same_sender_amount_perStep',
    'same_sender_type_recipient_perStep',
    'same_sender_recipient_amount_perStep',
    'same_sender_type_recipient_amount_perStep',
    'same_recipient_perStep',
    'same_recipient_type_perStep',
    'same_recipient_amount_perStep',
    'same_recipient_type_amount_perStep',
    'same_senderbal_txnamount',
    'senderbal_greaterthan10M',
]].to_numpy()

y_train = pre_df[["isFraud"]].to_numpy()

X_test = post_df[[
    'amount',
    # 'nameOrig',
    'oldbalanceOrig',
    'newbalanceOrig',
    # 'nameDest',
    'oldbalanceDest',
    'newbalanceDest',
    'type_IN',
    'type_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
    'same_sender_perStep',
    'same_sender_type_perStep',
    'same_sender_recipient_perStep',
    'same_sender_amount_perStep',
    'same_sender_type_recipient_perStep',
    'same_sender_recipient_amount_perStep',
    'same_sender_type_recipient_amount_perStep',
    'same_recipient_perStep',
    'same_recipient_type_perStep',
    'same_recipient_amount_perStep',
    'same_recipient_type_amount_perStep',
    'same_senderbal_txnamount',
    'senderbal_greaterthan10M',
]].to_numpy()

y_test= post_df[["isFraud"]].to_numpy()

# 1
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_lr))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is LogisticRegression')
acc= accuracy_score(y_test,y_pred_lr)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_lr)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_lr)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_lr)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_lr)
print('The Matthews correlation coefficient is{}'.format(MCC))

# [[1957032    1265]
#  [    831    3291]]
# The model used is LogisticRegression
# The accuracy is 0.9989319304389124
# The precision is 0.7223441615452151
# The recall is 0.7983988355167394
# The F1-Score is 0.7584696934777598
# The Matthews correlation coefficient is0.7588896659332206

# 2
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(confusion_matrix(y_test, y_pred_nb))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is GaussianNB')
acc= accuracy_score(y_test,y_pred_nb)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_nb)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_nb)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_nb)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_nb)
print('The Matthews correlation coefficient is{}'.format(MCC))

# [[1947046   11251]
#  [   3413     709]]
# The model used is GaussianNB
# The accuracy is 0.9925275896737649
# The precision is 0.0592809364548495
# The recall is 0.17200388161086852
# The F1-Score is 0.08817311279691581
# The Matthews correlation coefficient is0.09780101151457046

# 3
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print(confusion_matrix(y_test, y_pred_sgd))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is SGDClassifier')
acc= accuracy_score(y_test,y_pred_sgd)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_sgd)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_sgd)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_sgd)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_sgd)
print('The Matthews correlation coefficient is{}'.format(MCC))

# [[1950616    7681]
#  [   1214    2908]]
# The model used is SGDClassifier
# The accuracy is 0.9954673288426172
# The precision is 0.2746246104448012
# The recall is 0.705482775351771
# The F1-Score is 0.39535041805451704
# The Matthews correlation coefficient is0.4384397556635192

# 4
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred_knn))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is KNeighborsClassifier')
acc= accuracy_score(y_test,y_pred_knn)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_knn)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_knn)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_knn)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_knn)
print('The Matthews correlation coefficient is{}'.format(MCC))

#
#
#

# 5
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10, random_state=101,
                               max_features=None, min_samples_leaf=15
                              )
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
print(confusion_matrix(y_test, y_pred_dtree))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is DecisionTreeClassifier')
acc= accuracy_score(y_test,y_pred_dtree)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_dtree)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_dtree)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_dtree)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_dtree)
print('The Matthews correlation coefficient is{}'.format(MCC))

# array([[1958297,       0],
#        [      2,    4120]])

# 6
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(
    n_estimators=70, oob_score=True, n_jobs=-1,
    random_state=101, max_features=None, min_samples_leaf=30
                            )
rfm.fit(X_train, y_train)
y_pred_rfm = rfm.predict(X_test)
print(confusion_matrix(y_test, y_pred_rfm))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is RandomForestClassifier')
acc= accuracy_score(y_test,y_pred_rfm)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_rfm)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_rfm)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_rfm)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_rfm)
print('The Matthews correlation coefficient is{}'.format(MCC))

# 7
from sklearn.svm import SVC
svm = SVC(kernel="linear", C=0.025, random_state=101)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred_svm))

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
print('The model used is SVC')
acc= accuracy_score(y_test,y_pred_svm)
print('The accuracy is {}'.format(acc))
prec= precision_score(y_test,y_pred_svm)
print('The precision is {}'.format(prec))
rec= recall_score(y_test,y_pred_svm)
print('The recall is {}'.format(rec))
f1= f1_score(y_test,y_pred_svm)
print('The F1-Score is {}'.format(f1))
MCC=matthews_corrcoef(y_test,y_pred_svm)
print('The Matthews correlation coefficient is{}'.format(MCC))