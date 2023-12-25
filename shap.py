import xgboost

print("helicopter")

import pandas as pd
X_train = pd.read_csv("yourpath")
Y_train = pd.read_csv("yourpath")
X_test = pd.read_csv("yourpath")
Y_test = pd.read_csv("yourpath")

col = list(X_train)

from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=4000,
                         learning_rate=0.05,
                         subsample=1,
                         max_depth=6,
                         booster="gbtree")
testpred = xgb_model.fit(X_train, Y_train).predict(X_test)
print(testpred)
trainpred = xgb_model.fit(X_train, Y_train).predict(X_train)
print(trainpred)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
mestest = mean_squared_error(Y_test,testpred)
r2test = r2_score(Y_test,testpred)
maetest = mean_absolute_error(Y_test,testpred)
print(mestest)
print(r2test)
print(maetest)
mestrain = mean_squared_error(Y_train,trainpred)
r2train = r2_score(Y_train,trainpred)
maetrain = mean_absolute_error(Y_train,trainpred)
print(mestrain)
print(r2train)
print(maetrain)

import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
print(shap_values.shape)
y_base = explainer.expected_value
print(y_base)
shap.summary_plot(shap_values, X_train[col])
