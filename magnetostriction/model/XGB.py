
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

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

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param_grid = {
    'max_depth': [8],
    'min_child_weight': [3],
    'subsample': [0.7],
    'colsample_bytree': [0.1],
    'learning_rate': [0.5],
    'n_estimators': [2000],
    'objective': ['reg:squarederror'],
    # 'eval_metric': ['rmse']
    # 'max_leaves': [150]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# 计算RMSE和R²
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Best Parameters: {best_params}")
print(f"RMSE(Train): {rmse_train}", f"RMSE(Test): {rmse_test}")
print(f"R²(Train): {r2_train}", f"R²(Test): {r2_test}")

fig,ax = plt.subplots(figsize=(10, 8))
# best_line_x = np.linspace(300,1800)
# best_line_y = best_line_x
# best_line = ax.plot(best_line_x,best_line_y, c='k',linewidth=2,linestyle='--')

# plt.scatter(y_train, y_pred_train, alpha=0.7, s=50, c='#f08e64', edgecolors='w')
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', lw=2)
plt.scatter(y_test, y_pred_test, alpha=0.7, s=50, c='#64C0A6', edgecolors='w')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=3)
plt.text(400, 1600, 'R$^2$= %.2f'%r2_test, fontsize=23)
plt.xlim((300,1800))
plt.ylim((300,1800))
plt.tick_params(labelsize=23)
plt.tick_params(pad = 9)
plt.xlabel('Measured λa (ppm)',fontsize=25)
plt.ylabel('Predicted λa (ppm)',fontsize=25)

bwith = 3
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

# plt.title('True Values vs Predicted Values')
plt.grid(False)
plt.savefig('E:/python_code/magnetostriction/model/model_figure/XGB.png',
            bbox_inches='tight', dpi=600)
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

