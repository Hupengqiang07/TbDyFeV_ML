import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample

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

train_rmse_list = []
test_rmse_list = []

n_iterations = 100

# ///////////////////////////XGB model///////////////////////////////
# import xgboost as xgb
# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=i)
#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
#                                                         random_state=i + 1)

#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtest = xgb.DMatrix(X_test, label=y_test)

#     param_grid = {
#         'objective': ['reg:squarederror'],
#         'max_depth': [8],
#         'eta': [0.3],
#         'eval_metric': ['rmse']
#         }
#
#     grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid,
#                                scoring='neg_mean_squared_error', cv=10, verbose=1, n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     best_params = grid_search.best_params_
#     best_model = grid_search.best_estimator_

#     y_train_pred = best_model.predict(X_train)
#     y_test_pred = best_model.predict(X_test)

#     # 计算RMSE
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#
#     # 存储RMSE
#     train_rmse_list.append(train_rmse)
#     test_rmse_list.append(test_rmse)

# train_rmse_mean = np.mean(train_rmse_list)
# train_rmse_std = np.std(train_rmse_list)
# test_rmse_mean = np.mean(test_rmse_list)
# test_rmse_std = np.std(test_rmse_list)

# print(f"Bootstrap Training RMSE Mean: {train_rmse_mean}")
# print(f"Bootstrap Training RMSE Standard Deviation: {train_rmse_std}")
# print(f"Bootstrap Test RMSE Mean: {test_rmse_mean}")
# print(f"Bootstrap Test RMSE Standard Deviation: {test_rmse_std}")


# //////////////////////////////SVR model///////////////////////////

# from sklearn.svm import SVR

# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=i)
#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
#                                                         random_state=i + 1)

#     svr = SVR(kernel='rbf')

#     param_grid = {
#         'C': [1000],
#         'gamma': [0.3],
#         # 'epsilon': [0.1, 0.5, 1.0]
#     }

#     grid_search = GridSearchCV(svr, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     best_svr = grid_search.best_estimator_

#     y_train_pred = best_svr.predict(X_train)
#     y_test_pred = best_svr.predict(X_test)
#
#     # 计算RMSE
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#
#     # 存储RMSE
#     train_rmse_list.append(train_rmse)
#     test_rmse_list.append(test_rmse)

# train_rmse_mean = np.mean(train_rmse_list)
# train_rmse_std = np.std(train_rmse_list)
# test_rmse_mean = np.mean(test_rmse_list)
# test_rmse_std = np.std(test_rmse_list)

# print(f"Bootstrap Training RMSE Mean: {train_rmse_mean}")
# print(f"Bootstrap Training RMSE Standard Deviation: {train_rmse_std}")
# print(f"Bootstrap Test RMSE Mean: {test_rmse_mean}")
# print(f"Bootstrap Test RMSE Standard Deviation: {test_rmse_std}")


# //////////////////////////////RFR model///////////////////////////

# from sklearn.ensemble import RandomForestRegressor

# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=i)
#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
#                                                         random_state=i + 1)

#     rf = RandomForestRegressor(random_state=42)

#     param_grid = {
#         'n_estimators': [300],
#         'max_depth': [10],
#         'min_samples_split': [2],
#         'min_samples_leaf': [8]
#       }

#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2,
#                            n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     best_rf = grid_search.best_estimator_

#     y_train_pred = best_rf.predict(X_train)
#     y_test_pred = best_rf.predict(X_test)
#
#     # 计算RMSE
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#
#     # 存储RMSE
#     train_rmse_list.append(train_rmse)
#     test_rmse_list.append(test_rmse)

# train_rmse_mean = np.mean(train_rmse_list)
# train_rmse_std = np.std(train_rmse_list)
# test_rmse_mean = np.mean(test_rmse_list)
# test_rmse_std = np.std(test_rmse_list)

# print(f"Bootstrap Training RMSE Mean: {train_rmse_mean}")
# print(f"Bootstrap Training RMSE Standard Deviation: {train_rmse_std}")
# print(f"Bootstrap Test RMSE Mean: {test_rmse_mean}")
# print(f"Bootstrap Test RMSE Standard Deviation: {test_rmse_std}")

# ///////////////////////////knn model///////////////////////////////

# from sklearn.neighbors import KNeighborsRegressor

# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=i)
#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
#                                                         random_state=i + 1)

#     knn_regressor = KNeighborsRegressor()

#     param_grid = {'n_neighbors': [2]}

#     grid_search = GridSearchCV(knn_regressor, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2)
#     grid_search.fit(X_train, y_train)

#     best_knn = grid_search.best_estimator_

#     y_train_pred = best_knn.predict(X_train)
#     y_test_pred = best_knn.predict(X_test)
#
#     # 计算RMSE
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#
#     # 存储RMSE
#     train_rmse_list.append(train_rmse)
#     test_rmse_list.append(test_rmse)

# train_rmse_mean = np.mean(train_rmse_list)
# train_rmse_std = np.std(train_rmse_list)
# test_rmse_mean = np.mean(test_rmse_list)
# test_rmse_std = np.std(test_rmse_list)

# print(f"Bootstrap Training RMSE Mean: {train_rmse_mean}")
# print(f"Bootstrap Training RMSE Standard Deviation: {train_rmse_std}")
# print(f"Bootstrap Test RMSE Mean: {test_rmse_mean}")
# print(f"Bootstrap Test RMSE Standard Deviation: {test_rmse_std}")

# ///////////////////////////DTR model///////////////////////////////

# from sklearn.tree import DecisionTreeRegressor

# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X), random_state=i)
#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
#                                                         random_state=i + 1)

#     param_grid = {
#         'max_depth': [16],
#         'min_samples_split': [6],
#         'min_samples_leaf': [3]
#     }

#     regressor = DecisionTreeRegressor(random_state=42)

#     grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error',
#                            verbose=1, n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     best_dt = grid_search.best_estimator_

#     y_train_pred = best_dt.predict(X_train)
#     y_test_pred = best_dt.predict(X_test)
#
#     # 计算RMSE
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#
#     # 存储RMSE
#     train_rmse_list.append(train_rmse)
#     test_rmse_list.append(test_rmse)

# train_rmse_mean = np.mean(train_rmse_list)
# train_rmse_std = np.std(train_rmse_list)
# test_rmse_mean = np.mean(test_rmse_list)
# test_rmse_std = np.std(test_rmse_list)

# print(f"Bootstrap Training RMSE Mean: {train_rmse_mean}")
# print(f"Bootstrap Training RMSE Standard Deviation: {train_rmse_std}")
# print(f"Bootstrap Test RMSE Mean: {test_rmse_mean}")
# print(f"Bootstrap Test RMSE Standard Deviation: {test_rmse_std}")


