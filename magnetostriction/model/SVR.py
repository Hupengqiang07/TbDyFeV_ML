import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_excel('TbDyFe database.xlsx')
# print(data)

dataset = pd.DataFrame(data, columns=['Tb','Dy','Pr','Nd','Ho','Fe','Co','Cu','V','Cr','B','Al','Ga','Be','Si',
                                      'r_emp.2','Tm_ave.','Tm_emp.1','Tb_ave.','Tb_emp.1','Hf_ave.','Hf_emp.1','Hv_ave.',
                                      'Hv_emp.1','Hs_ave.','Hs_emp.1','k_ave.','k_emp.1','μ_ave.','μ_emp.1','Vm_ave.',
                                      'Vm_emp.1','Mb_ave.','Mb_emp.2','Ms_ave.','Ms_emp.2','My_ave.','My_emp.2','En_ave.',
                                      'En_emp.1','VEC_ave.','VEC_emp.1','Ea_ave.','Ea_emp.2','Ei_ave.','Ei_emp.2','T_heat',
                                      't_heat','H_cri.','λa'])

X = dataset.loc[:, ['Tb','Dy','Pr','Nd','Ho','Fe','Co','Cu','V','Cr','B','Al','Ga','Be','Si',
                    'Tb_ave.','Hs_ave.','μ_ave.','Mb_ave.','VEC_emp.1','Ei_ave.','T_heat','H_cri.']].values
y = dataset.loc[:, 'λa'].values

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr = SVR(kernel='rbf')

param_grid = {
    'C': [1000,10000,100000,150000],
    'gamma': [0.001,0.01,0.1,0.3,0.5],
    # 'epsilon': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(svr, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_svr = grid_search.best_estimator_

y_pred_train = best_svr.predict(X_train)
y_pred_test = best_svr.predict(X_test)

# 计算RMSE和R²
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"RMSE(Train): {rmse_train}", f"RMSE(Test): {rmse_test}")
print(f"R²(Train): {r2_train}", f"R²(Test): {r2_test}")

# mae = mean_absolute_error(y_test, y_pred)

plt.figure(figsize=(10, 8))

# plt.scatter(y_train, y_pred_train, alpha=0.7, s=50, c='#8C9FCA', edgecolors='w')
plt.scatter(y_test, y_pred_test, alpha=0.7, s=50, c='#64C0A6', edgecolors='w')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.tick_params(labelsize=23)
plt.tick_params(pad = 0.9)
plt.xlabel('Measured λa (ppm)',fontsize=23)
plt.ylabel('Predicted λa (ppm)',fontsize=23)

# plt.title('True Values vs Predicted Values')
plt.grid(False)
# plt.savefig('E:/python_code/magnetostriction/model/model_figure/SVR.png',
#             bbox_inches='tight', dpi=600)
plt.show()

residuals = y_test - y_pred_test
plt.figure(figsize=(10, 8))
plt.scatter(y_pred_test, residuals, alpha=0.7, s=50, c='b', edgecolors='w')
plt.tick_params(labelsize=23)
plt.axhline(y=0, color='r', lw=2)
plt.xlabel('Predicted λa (ppm)',fontsize=23)
plt.ylabel('Residuals',fontsize=23)
plt.title('Residuals vs Predicted λa')
plt.grid(False)
# plt.savefig('E:/python_code/magnetostriction/model/model_figure/SVR_Residuals.png',
#             bbox_inches='tight', dpi=600)
# plt.show()

