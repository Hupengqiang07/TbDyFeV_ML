import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100,200,300,500,800],  # 森林中树的数量
    'max_depth': [5,10,15,20],  # 树的最大深度
    'min_samples_split': [2,5,8,12],  # 分割内部节点所需的最小样本数
    'min_samples_leaf': [2,5,8,12]  # 叶节点必须具有的最小样本数
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

# 计算RMSE和R²
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"RMSE(Train): {rmse_train}", f"RMSE(Test): {rmse_test}")
print(f"R²(Train): {r2_train}", f"R²(Test): {r2_test}")

fig,ax = plt.subplots(figsize=(10, 8))
# best_line_x = np.linspace(300,1800)
# best_line_y = best_line_x
# best_line = ax.plot(best_line_x,best_line_y, c='k',linewidth=2,linestyle='--')

plt.scatter(y_train, y_pred_train, alpha=0.7, s=50, c='#f08e64', edgecolors='w')
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', lw=2)
plt.scatter(y_test, y_pred_test, alpha=0.7, s=50, c='#64C0A6', edgecolors='w')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=3)
#设置坐标轴范围
plt.xlim((300,1800))
plt.ylim((300,1800))
plt.tick_params(labelsize=23)
plt.tick_params(pad = 0.9)
plt.xlabel('Measured λa (ppm)',fontsize=23)
plt.ylabel('Predicted λa (ppm)',fontsize=23)

# plt.title('True Values vs Predicted Values')
plt.grid(False)
# plt.savefig('E:/python_code/magnetostriction/model/model_figure/XGB.png',
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
# plt.savefig('E:/python_code/magnetostriction/model/model_figure/XGB_Residuals.png',
#             bbox_inches='tight', dpi=600)
# plt.show()

