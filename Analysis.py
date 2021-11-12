
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier


DF=pd.read_csv("data_interview_test.csv",delimiter=":")

DF["Matched"]=np.where(DF.matched_transaction_id==DF.feature_transaction_id,0,1)

DF["feature_transaction_id"]=DF["feature_transaction_id"].str.replace(',', '').astype("int64")

DF["receipt_id"]=DF["receipt_id"].str.replace(',', '').astype("int64")

DF["matched_transaction_id"]=DF["matched_transaction_id"].str.replace(',', '').astype("int64")

X=DF.iloc[:,0:-1]
y=DF.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())



Model=CatBoostClassifier(iterations=60)

Model_Unbalanced=CatBoostClassifier(iterations=60)

Model_Unbalanced.fit(X_train,y_train,eval_set=(X_test,y_test))

Model.fit(X_train_res,y_train_res,eval_set=(X_test,y_test))

Predictions_Prob=Model.predict_proba(X_test)

Predictions=Model.predict(X_test)

Prediction_Probability=[x[0] for x in Predictions_Prob]

Final_TEST=X_test.copy()

Final_TEST["Prediction_Probability"]=Prediction_Probability

Final_TEST.head(5)

Predictions_Unbalanced=Model_Unbalanced.predict(X_test)

# F1 score and confusion matrix to evaluate the model performance

from sklearn.metrics import confusion_matrix

cf_matrix=confusion_matrix(Predictions, y_test)

cf_matrix

import seaborn as sns
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

from sklearn.metrics import precision_recall_fscore_support as score



precision, recall, fscore, support = score(y_test, Predictions)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# Evaluation for data imbalance model

cf_matrix=confusion_matrix(Predictions_Unbalanced, y_test)

print(cf_matrix)

import seaborn as sns
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

precision, recall, fscore, support = score(y_test, Predictions_Unbalanced)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# We define a function which can attach itself to a model and then make predictions on a dataframe passed to it
# This function would then sort the dataset on "matched transaction id" and then return probabilities of it being a correct match

# Here we attach a model which would be used at the user-defined function below
MODEL= Model_Unbalanced

def Predict_Probability(data):
  Predictions_new=MODEL.predict_proba(data)
  Prediction_Probability=[x[0] for x in Predictions_new]
  data["Prediction_Probabilities"]=Prediction_Probability
  data=data.sort_values(by=['matched_transaction_id','Prediction_Probabilities'], ascending=[True,False])

  #print(Predictions_new)
  return data

Simulated_data=X_test.iloc[10:15,:]



Final_dataframe=Predict_Probability(Simulated_data)

print(Final_dataframe)

