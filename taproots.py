__author__ = 'Raihan Masud'

import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

"""
open source
work done for non profit taproots
DataDive, DataKind , Los Angeles
"""

#region data prep
def read_proj_application_data(file):
    print("loading data...")
    data = pd.read_csv(file, encoding='ISO-8859-1', error_bad_lines=False)
    data.drop(data.columns[[3, 8]], axis=1, inplace=True)  # drop application dates etc
    data.drop(data.columns[[22, 60]], axis=1, inplace=True)  # drop budget data
    return data


def read_proj_award_data(file):
    data = pd.read_csv(file, encoding='ISO-8859-1')
    return data


p_app_path = "./CSV/Projects_Applications.csv"
proj_app_data = read_proj_application_data(p_app_path)
print(proj_app_data.shape)

p_award_path = "./CSV/Projects_Awarded.csv"
proj_award_data = read_proj_award_data(p_award_path)
print(proj_award_data.shape)

app_award_data = pd.merge(proj_app_data, proj_award_data, how='inner', on=['sgapp_id', 'org_id'])
print(app_award_data.shape)
#print(app_award_data['project_status'])
#endregion data prep

#region visualization
#grant types applications-awards
app_type_df = pd.DataFrame(proj_app_data.assigned_grant_type.value_counts())
app_type_df.columns = ['Number of applications']
sp_ax = plt.subplot2grid((1, 4), (0, 1), colspan=3)
app_type_bar = app_type_df.plot(ax=sp_ax, kind='barh', legend=False)
app_type_bar.invert_yaxis()
app_type_bar.set_xlabel("Number of applications", fontsize="10")
#plt.show() enable this line to plot

sp_ax1 = plt.subplot2grid((1, 4), (0, 1), colspan=3)
awrd_type_df = pd.DataFrame(proj_award_data.grant_type.value_counts(), columns=['Number of applications'])
awrd_type_df.columns = ['Number of awards']
awrd_type_bar = awrd_type_df.plot(ax=sp_ax1, kind='barh', legend=False)
awrd_type_bar.invert_yaxis()
awrd_type_bar.set_xlabel("Number of awards", fontsize="10")
#plt.show() enable this line to plot

#areas non profit
sp_ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)
app_area_df = pd.DataFrame(proj_app_data.issue_area.value_counts())
app_area_bar = app_area_df.plot(ax=sp_ax2, kind='barh', legend=False)
app_area_bar.invert_yaxis()
app_area_bar.set_xlabel("Number of applications", fontsize="10")
#plt.show() enable this line to plot

sp_ax3 = plt.subplot2grid((1, 4), (0, 1), colspan=3)
awrd_area_df = pd.DataFrame(proj_award_data.issue_area.value_counts(), columns=['Number of awards'])
awrd_area_df.columns = ['Number of awards']
awrd_area_bar = awrd_area_df.plot(ax=sp_ax3, kind='barh', legend=False)
awrd_area_bar.invert_yaxis()
awrd_area_bar.set_xlabel("Number of awards", fontsize="10")
#plt.show() enable this line to plot
#non-profit areas
#endregion visualization

#ToDo: Model Needs work
#region building model to select features and predict what applications/non-profits are more likely to complete project/successful
#model of those awarded who did complete
#app_award_data.drop(app_award_data.columns[[0, 10]], axis=1, inplace=True)

#app_award_data.drop(app_award_data.columns[[0, 8]], axis=1, inplace=True)
print(app_award_data.columns.values)
print(app_award_data.shape)

app_award_data_clean = app_award_data[app_award_data['project_status'] != 'Not a Project']
print(app_award_data_clean.shape)
app_award_data_clean = app_award_data_clean[app_award_data_clean['project_status'] != 'In Progress']
print(app_award_data_clean.shape)
app_award_data_clean = app_award_data_clean[app_award_data_clean['project_status'] != 'On Hold']

#app_award_data_clean = app_award_data_clean.fillna(0)

X = app_award_data_clean[
    [  'employees_ft', 'employees_pt', 'volunteers_num_annual', 'constituents',
       'recommend_taproot', 'recommend_pro_bono']]

X.to_csv('./x_data.csv')

"""
'have_strategic_plan',
'issue_area'
'location_name',
'alert_level'
'region_name',
'app_assessment',
'assigned_grant_type',
"""

Y = app_award_data_clean['project_status']
Y[Y=='Complete'] = 1
Y[Y=='Released'] = 0

#RandomForest
clf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                        random_state=0, max_features="auto")
#impute data
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
"""
Xd = pd.DataFrame(columns=['employees_ft', 'employees_pt', 'volunteers_num_annual', 'constituents',
       'recommend_taproot', 'recommend_pro_bono'], data=X)
Xd.to_csv('./x_data_transformed.csv')
"""

"""
clf_rf = Pipeline([("imputer", Imputer(missing_values='NaN', strategy="mean",axis=0)),
                   ("forest", clf)])
"""

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, Y, test_size=0.30, random_state=0)

scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, scoring='mean_absolute_error')
print("CV score: ", abs(sum(scores) / len(scores)))

print("training model...")
#model = clf.fit(train_input, labels)
model = clf.fit(X_train, y_train)

#feature engineering
print("feature selection...")
print("feature importance", model.feature_importances_)  #get the values of columns
#['employees_ft', 'employees_pt', 'volunteers_num_annual', 'constituents','recommend_taproot', 'recommend_pro_bono']
#feature importance [ 0.29520347  0.14368023  0.11865674  0.40549507  0.01805286  0.01891162]

print("testing on holdout set...")
pred_y = model.predict(X_test)

print("Mean Absolute Error", mean_absolute_error(y_test, pred_y))

#endregion Model