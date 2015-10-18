__author__ = 'Raihan Masud'

import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import ensemble

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
proj_awad_data = read_proj_award_data(p_award_path)
print(proj_awad_data.shape)

app_award_data = pd.merge(proj_app_data, proj_awad_data, how='inner', on=['sgapp_id', 'org_id'])
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
awrd_type_df = pd.DataFrame(proj_awad_data.grant_type.value_counts(), columns=['Number of applications'])
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
awrd_area_df = pd.DataFrame(proj_awad_data.issue_area.value_counts(), columns=['Number of awards'])
awrd_area_df.columns = ['Number of awards']
awrd_area_bar = awrd_area_df.plot(ax=sp_ax3, kind='barh', legend=False)
awrd_area_bar.invert_yaxis()
awrd_area_bar.set_xlabel("Number of awards", fontsize="10")
#plt.show() enable this line to plot
#non-profit areas
#endregion visualization
#ToDo: Model Need work
#region building model to select features and predict what applications/non-profits are more likely to complete project/successful
#model of those awarded who did complete
#app_award_data.drop(app_award_data.columns[[0, 10]], axis=1, inplace=True)

#app_award_data.drop(app_award_data.columns[[0, 8]], axis=1, inplace=True)
#print(app_award_data.columns.values)
app_award_data_clean = app_award_data[app_award_data['project_status'] != 'Not a Project']

app_award_data_clean = app_award_data_clean.fillna(0)

X = app_award_data_clean[
    [  'employees_ft', 'employees_pt', 'volunteers_num_annual', 'constituents',
       'recommend_taproot', 'recommend_pro_bono']]
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


clf_rf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                        random_state=0, max_features="auto")

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, Y, test_size=0.30, random_state=0)

#scores = cross_validation.cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='mean_absolute_error')
#print("CV score: ", abs(sum(scores) / len(scores)))

print("training model...")
#model = clf.fit(train_input, labels)
model = clf_rf.fit(X_train, y_train)

#feature engineering
print("feature selection...")
print("feature importance", model.feature_importances_)  #get the values of columns

print("testing on holdout set...")
pred_y = model.predict(X_test)

print("Mean Absolute Error", mean_absolute_error(y_test, pred_y))

#endregion Model