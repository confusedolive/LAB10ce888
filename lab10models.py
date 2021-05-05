import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
print(np.__version__)
model_dir = os.path.join(os.getcwd(), 'models')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

path = r'C:\Users\jeron\OneDrive\Desktop\lab10\Lab10\data\heart.csv'
data = pd.read_csv(path)

data.head()
data.isnull().sum()

X = data.iloc[:, :-1].copy()
y = data.iloc[:, -1].copy()

standard = StandardScaler()
to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
standarizeed = standard.fit_transform(X.values)

X[X.columns] = standarizeed


def boruta_selected():
    randomclf = RandomForestClassifier(n_jobs=-1,
                                       max_depth=6, n_estimators=1000,
                                       class_weight='balanced')

    boruta_select = BorutaPy(randomclf, n_estimators='auto',
                             verbose=2, random_state=42)

    boruta_select.fit(np.array(X), np.array(y))

    features_importance = [X.columns[i]
                           for i, boolean in enumerate(boruta_select.support_) if boolean]

    not_important = [X.columns[i]
                     for i, boolean in enumerate(boruta_select.support_) if not boolean]
    return features_importance, not_important


features_importance, not_importante = boruta_selected()


X = X[columns_important]

columns_important = ['age', 'cp', 'thalach', 'exang', 'oldpeak' ,'ca' ,'thal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
##################logistic regression #######################################
data.thal.min()
data.thal.max()
lr = LogisticRegression()
params = [
    {'penalty': ['l1'], 'C': np.logspace(-4, 4, 50),
        'solver': ['liblinear', 'saga'],
     'max_iter':[100, 1000, 1500, 5000]},
    {'penalty': ['l2'], 'C': np.logspace(-4, 4, 50),
     'solver': ['newton-cg', 'sag', 'lbfgs'],
     'max_iter':[100, 1000, 1500, 5000]},
    {'penalty': ['elasticnet'], 'C': np.logspace(-4, 4, 20),
     'solver': ['saga'], 'max_iter':[100, 1000, 1500, 5000],
     'l1_ratio':[0.5]}]

lr.fit(X_train, y_train)
lr.score(X_test, y_test)

cv_grid_lr = GridSearchCV(estimator=lr, param_grid=params, refit=True)
cv_grid_lr.fit(X_train, y_train)

cv_grid_lr.score(X_test, y_test)

pkl_file = os.path.join(model_dir, 'logistic_pkl_model.pkl')

with open(pkl_file, 'wb') as file:
    pkl.dump(cv_grid_lr, file)

##############random forest#######################
rf = RandomForestClassifier()
params = {'n_estimators': [100, 200, 500, 1000], 'max_depth': [4, 5, 6, 8, 12]}

cv_grid = GridSearchCV(estimator=rf, param_grid=params, cv=5, refit=True)
cv_grid.fit(X_train, y_train)
rf.fit(X_train,y_train)
print(cv_grid.best_estimator_)
cv_grid.score(X_test, y_test)
print(rf.score(X_test,y_test))
pkl_random = os.path.join(model_dir, 'rf_pkl_model.pkl')

with open(pkl_random, 'wb') as file:
    pkl.dump(cv_grid, file)
#################SVC###################

svc = SVC()
params = {'C': [0.001, 0.01, 0.1, 1, 10],
          'gamma': [0.001, 0.01, 0.1, 1]}

cv_grid_svc = GridSearchCV(estimator=svc, param_grid=params, cv=5, refit=True)
cv_grid_svc.fit(X_train,  y_train)

cv_grid_svc.score(X_test, y_test)

pkl_svc = os.path.join(model_dir, 'svc_pkl_model.pkl')

with open(pkl_svc, 'wb') as f:
    pkl.dump(cv_grid_svc, f)

with open('models/logistic_pkl_model.pkl', 'rb') as f:
    logistic = pkl.load(f)
logistic.predict(X)[0]
