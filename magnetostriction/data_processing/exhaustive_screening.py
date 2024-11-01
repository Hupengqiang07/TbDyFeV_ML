import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations
import warnings

data = pd.read_excel('TbDyFe database.xlsx')
# print(data)

dataset = pd.DataFrame(data, columns=['r_emp.2','Tm_ave.','Tm_emp.1','Tb_ave.','Tb_emp.1','Hf_ave.','Hf_emp.1','Hv_ave.',
                                      'Hv_emp.1','Hs_ave.','Hs_emp.1','k_ave.','k_emp.1','μ_ave.','μ_emp.1','Vm_ave.',
                                      'Vm_emp.1','Mb_ave.','Mb_emp.2','Ms_ave.','Ms_emp.2','My_ave.','My_emp.2','En_ave.',
                                      'En_emp.1','VEC_ave.','VEC_emp.1','Ea_ave.','Ea_emp.2','Ei_ave.','Ei_emp.2','T_heat',
                                      't_heat','H_cri.','λa'])

X = dataset.loc[:, ['Tb_ave.','Tb_emp.1','Hv_ave.','Hs_ave.','μ_ave.','Vm_ave.','Mb_ave.','VEC_emp.1','Ea_emp.2',
                    'Ei_ave.','T_heat','H_cri.']].values
y = dataset.loc[:, 'λa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# param_grid = {
#     'n_estimators': [500],
#     'max_depth': [8],
#     'learning_rate': [0.04]
# }

results = []

for r in range(1, X.shape[1] + 1):
    for combo in combinations(range(X.shape[1]), r):
        X_train_combo = X_train[:, list(combo)]
        X_test_combo = X_test[:, list(combo)]

        model = xgb.XGBRegressor(n_estimators=800,max_depth=8,learning_rate=0.5, random_state=0)
        model.fit(X_train_combo, y_train)

        y_pred = model.predict(X_test_combo)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'feature_combination': list(combo),
            'rmse': rmse
        })

results_df = pd.DataFrame(results)
results_df.to_csv('E:/python_code/magnetostriction/data_processing/ES_3fold_2.csv', index=False)
print("Results saved to 'xgboost_feature_combination_results.csv'")

