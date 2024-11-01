import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import shap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore")

data = pd.read_excel('SHAP_TbDyFe database.xlsx')
# print(data)

X= data.drop('λa',axis=1)
y= data['λa']
# print(y)
# X = pd.DataFrame(data, columns=['Tb_ave.','Hs_ave.','μ_ave.','Mb_ave.','VEC_emp.1','Ei_ave.','T_heat','H_cri.'])
# y = pd.DataFrame(data, columns=['λa'])
# data_array = data.to_numpy()
# print(data_array)

# X = data['Tb_ave.','Hs_ave.','μ_ave.','Mb_ave.','VEC_emp.1','Ei_ave.','T_heat','H_cri.'])
# y = pd.DataFrame(data, columns=['λa'])
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
data.head()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 8,
    'eta': 0.5,
    'objective': 'reg:squarederror',
    'n_estimators': 800
}

best_model = xgb.train(params, dtrain, num_boost_round=671)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(dtest)

shap.initjs()

# summarize the effects of all the features

# mpl.rcParams['axes.linewidth'] = 2.0
# mpl.rcParams['font.size'] = 50
# mpl.rcParams['xtick.labelsize'] = 20
# mpl.rcParams['ytick.labelsize'] = 20

shap.summary_plot(shap_values, X_test, show=False)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.xlabel('SHAP value (impact on model output)', fontsize=20)

plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/shap-summary.png', bbox_inches='tight', dpi=600)
plt.show()

fig = plt.subplots(figsize=(8,10))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.xlabel('mean (|SHAP value|)(average impact on \n model output magnitude', fontsize=20)
# plt.ylabel('SHAP value for Hs_ave.', fontsize=28)
plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/shap-bar.png', bbox_inches='tight', dpi=600)
plt.show()

shap.force_plot(explainer.expected_value, shap_values,X_test,show=False)
plt.show()
#
fig = plt.subplots(figsize=(10,8))
shap_values=explainer(X[:3000])
shap.plots.heatmap(shap_values,max_display=8,show=False)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.xlabel('Instances', fontsize=20)
plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/shap-heatmap.png', bbox_inches='tight', dpi=600)
plt.show()

