import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

data = pd.read_excel('TbDyFe database.xlsx')
# print(data)

dataset = pd.DataFrame(data, columns=['r_emp.2','Tm_ave.','Tm_emp.1','Tb_ave.','Tb_emp.1','Hf_ave.','Hf_emp.1','Hv_ave.',
                                      'Hv_emp.1','Hs_ave.','Hs_emp.1','k_ave.','k_emp.1','μ_ave.','μ_emp.1','Vm_ave.',
                                      'Vm_emp.1','Mb_ave.','Mb_emp.2','Ms_ave.','Ms_emp.2','My_ave.','My_emp.2','En_ave.',
                                      'En_emp.1','VEC_ave.','VEC_emp.1','Ea_ave.','Ea_emp.2','Ei_ave.','Ei_emp.2','T_heat',
                                      't_heat','H_cri.','λa'])

X = dataset.loc[:, ['r_emp.2','Tm_ave.','Tm_emp.1','Tb_ave.','Tb_emp.1','Hf_ave.','Hf_emp.1','Hv_ave.',
                                      'Hv_emp.1','Hs_ave.','k_ave.','k_emp.1','μ_ave.','Vm_ave.',
                                      'Vm_emp.1','Mb_ave.','Mb_emp.2','Ms_ave.','Ms_emp.2','En_ave.',
                                      'En_emp.1','VEC_emp.1','Ea_ave.','Ea_emp.2','Ei_ave.','Ei_emp.2','T_heat',
                                      't_heat','H_cri.']].values
y = dataset.loc[:, 'λa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [5,10,15,20],
    'min_samples_split': [2,5,8,12],
    'min_samples_leaf': [1,3,5,8,12]
}
grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters：", grid_search.best_params_)

thresholds = np.linspace(0, 0.05, 26)
rmse_scores = []

for threshold in thresholds:
    selector = SelectFromModel(best_rf, threshold=threshold)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    rf_selected = RandomForestRegressor(random_state=42, **grid_search.best_params_)
    rf_selected.fit(X_train_selected, y_train)

    y_pred = rf_selected.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    print(f"threshold: {threshold:.5f}, number of feature: {X_train_selected.shape[1]}, RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, rmse_scores, marker='o')
plt.xlabel('Feature Importance Threshold')
plt.ylabel('RMSE')
plt.title('Threshold vs RMSE')
plt.grid(True)
plt.show()

best_threshold = thresholds[np.argmin(rmse_scores)]
best_selector = SelectFromModel(best_rf, threshold=best_threshold)
X_train_best = best_selector.fit_transform(X_train, y_train)

print("Best threshold：", best_threshold)
print("Selected feature index：", np.where(best_selector.get_support())[0])
