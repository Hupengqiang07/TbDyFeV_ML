import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore")

data = pd.read_excel('TbDyFe database.xlsx')

X = pd.DataFrame(data, columns=['Tb','Fe','V','Tb_ave.','Hs_ave.','μ_ave.','Mb_ave.','VEC_emp.1','Ei_ave.','T_heat','H_cri.'])
y = pd.DataFrame(data, columns=['λa'])

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

# 训练模型
best_model = xgb.train(params, dtrain, num_boost_round=671)

import shap

explainer = shap.TreeExplainer(best_model)
# 计算shap值为numpy.array数组
shap_values_numpy = explainer.shap_values(X)

shap_values_data = pd.DataFrame(shap_values_numpy, columns=X.columns)
print(shap_values_data.head())

# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['μ_emp.1'], shap_values_data['μ_emp.1'], s=10)

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=1)
# plt.xlabel('μ_emp.1', fontsize=12)
# plt.ylabel('SHAP value for μ_emp.1', fontsize=12)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # plt.savefig("SHAP Dependence Plot_1.pdf", format='pdf', bbox_inches='tight')
# plt.show()

import seaborn as sns
from scipy.optimize import fsolve

# /////////////////////////// μ_ave //////////////////////////////////

# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['μ_ave.'], shap_values_data['μ_ave.'], s=25, c='#6E8FB2', label='SHAP values', alpha=0.7)
#
# sns.regplot(x=data['μ_ave.'], y=shap_values_data['μ_ave.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['μ_ave.'], y=shap_values_data['μ_ave.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at μ_ave. = {x_intercept:.3f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')

# plt.legend(loc="lower right",prop = {'size':20})

# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('μ_ave.', fontsize=28)
# plt.ylabel('SHAP value for μ_ave.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
# # plt.savefig("SHAP Dependence Plot_with_Multiple_Intersections.pdf", format='pdf', bbox_inches='tight')
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/μ_ave.png', bbox_inches='tight', dpi=600)
# plt.show()

# /////////////////////////// Mb_ave. //////////////////////////////////
# 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['Mb_ave.'], shap_values_data['Mb_ave.'], s=25, c='#7DA494', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['Mb_ave.'], y=shap_values_data['Mb_ave.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['Mb_ave.'], y=shap_values_data['Mb_ave.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)
#
# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Mb_ave. = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')

# plt.legend(loc="upper left",prop = {'size':17})

# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('Mb_ave.', fontsize=28)
# plt.ylabel('SHAP value for Mb_ave.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Mb_ave.png', bbox_inches='tight', dpi=600)
# plt.show()
#
# # /////////////////////////// Ei_ave. //////////////////////////////////
# 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['Ei_ave.'], shap_values_data['Ei_ave.'], s=25, c='#E5A79A', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['Ei_ave.'], y=shap_values_data['Ei_ave.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['Ei_ave.'], y=shap_values_data['Ei_ave.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Ei_ave. = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')
#
# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="lower right",prop = {'size':16})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('Ei_ave.', fontsize=28)
# plt.ylabel('SHAP value for Ei_ave.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
# # plt.savefig("SHAP Dependence Plot_with_Multiple_Intersections.pdf", format='pdf', bbox_inches='tight')
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Ei_ave.png', bbox_inches='tight', dpi=600)
# plt.show()

# # /////////////////////////// T_heat //////////////////////////////////
# 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['T_heat'], shap_values_data['T_heat'], s=25, c='#EAB67A', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['T_heat'], y=shap_values_data['T_heat'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['T_heat'], y=shap_values_data['T_heat'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at T_heat = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="upper left",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('T_heat', fontsize=28)
# plt.ylabel('SHAP value for T_heat', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/T_heat.png', bbox_inches='tight', dpi=600)
# plt.show()
#
# # /////////////////////////// H_cri. //////////////////////////////////
# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['H_cri.'], shap_values_data['H_cri.'], s=25, c='#9F8DB8', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['H_cri.'], y=shap_values_data['H_cri.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['H_cri.'], y=shap_values_data['H_cri.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at H_cri. = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="upper right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('H_cri.', fontsize=28)
# plt.ylabel('SHAP value for H_cri.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/H_cri.png', bbox_inches='tight', dpi=600)
# plt.show()
#
# # /////////////////////////// Tb_ave. //////////////////////////////////
# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['Tb_ave.'], shap_values_data['Tb_ave.'], s=25, c='#ABC8E5', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['Tb_ave.'], y=shap_values_data['Tb_ave.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['Tb_ave.'], y=shap_values_data['Tb_ave.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Tb_ave. = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="lower right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('Tb_ave.', fontsize=28)
# plt.ylabel('SHAP value for Tb_ave.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Tb_ave.png', bbox_inches='tight', dpi=600)
# plt.show()
#
# # /////////////////////////// Hs_ave. //////////////////////////////////
# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['Hs_ave.'], shap_values_data['Hs_ave.'], s=25, c='#D8A0C1', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['Hs_ave.'], y=shap_values_data['Hs_ave.'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['Hs_ave.'], y=shap_values_data['Hs_ave.'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings
#
# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Hs_ave. = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="lower right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('Hs_ave.', fontsize=28)
# plt.ylabel('SHAP value for Hs_ave.', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Hs_ave.png', bbox_inches='tight', dpi=600)
# plt.show()
#
# # /////////////////////////// VEC_emp.1 //////////////////////////////////
# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['VEC_emp.1'], shap_values_data['VEC_emp.1'], s=25, c='#D0D08A', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['VEC_emp.1'], y=shap_values_data['VEC_emp.1'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['VEC_emp.1'], y=shap_values_data['VEC_emp.1'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at VEC_emp.1 = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="upper right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('VEC_emp.1', fontsize=28)
# plt.ylabel('SHAP value for VEC_emp.1', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/VEC_emp.1.png', bbox_inches='tight', dpi=600)
# plt.show()

# # /////////////////////////// Tb content //////////////////////////////////
# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['Tb'], shap_values_data['Tb'], s=25, c='#487DB2', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['Tb'], y=shap_values_data['Tb'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['Tb'], y=shap_values_data['Tb'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)
#
# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Tb = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="lower right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('Tb', fontsize=28)
# plt.ylabel('SHAP value for Tb', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Tb.png', bbox_inches='tight', dpi=600)
# plt.show()

# # /////////////////////////// Fe content //////////////////////////////////
# # 绘制散点图
plt.figure(figsize=(10, 8), dpi=600)
plt.scatter(data['Fe'], shap_values_data['Fe'], s=25, c='#826BA2', label='SHAP values', alpha=0.7)

sns.regplot(x=data['Fe'], y=shap_values_data['Fe'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

lowess_data = sns.regplot(x=data['Fe'], y=shap_values_data['Fe'], scatter=False, lowess=True, color='r')
line = lowess_data.get_lines()[0]
x_fit = line.get_xdata()
y_fit = line.get_ydata()

def find_zero_crossings(x_fit, y_fit):
    crossings = []
    for i in range(1, len(y_fit)):
        if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):

            crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
            crossings.append(crossing)
    return crossings

x_intercepts = find_zero_crossings(x_fit, y_fit)

for x_intercept in x_intercepts:
    plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at Fe = {x_intercept:.2f}')
    # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')

# 添加图例
plt.legend(loc="upper left",prop = {'size':18})

# 设置标签和标题
plt.xticks(size = 25)
plt.yticks(size = 25)
plt.xlabel('Fe', fontsize=28)
plt.ylabel('SHAP value for Fe', fontsize=28)

bwith = 3
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)#图框下边
ax.spines['left'].set_linewidth(bwith)#图框左边
ax.spines['top'].set_linewidth(bwith)#图框上边
ax.spines['right'].set_linewidth(bwith)#图框右边
# 保存并显示图像

plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/Fe.png', bbox_inches='tight', dpi=600)
plt.show()
#
# # # /////////////////////////// V content //////////////////////////////////
# # # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['V'], shap_values_data['V'], s=25, c='#CA81A9', label='SHAP values', alpha=0.7)

# sns.regplot(x=data['V'], y=shap_values_data['V'], scatter=False, lowess=True, color='r', label='LOWESS Curve')

# lowess_data = sns.regplot(x=data['V'], y=shap_values_data['V'], scatter=False, lowess=True, color='r')
# line = lowess_data.get_lines()[0]
# x_fit = line.get_xdata()
# y_fit = line.get_ydata()

# def find_zero_crossings(x_fit, y_fit):
#     crossings = []
#     for i in range(1, len(y_fit)):
#         if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#
#             crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#             crossings.append(crossing)
#     return crossings

# x_intercepts = find_zero_crossings(x_fit, y_fit)

# for x_intercept in x_intercepts:
#     plt.axvline(x=x_intercept, color='blue', linestyle='--', linewidth=3, label=f'Intersection at V = {x_intercept:.2f}')
#     # plt.text(x_intercept, 0.2, f'μ_emp.1 = {x_intercept:.2f}', color='blue', fontsize=10, verticalalignment='bottom')

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=3, label='SHAP = 0')
#
# # 添加图例
# plt.legend(loc="lower right",prop = {'size':18})
#
# # 设置标签和标题
# plt.xticks(size = 25)
# plt.yticks(size = 25)
# plt.xlabel('V', fontsize=28)
# plt.ylabel('SHAP value for V', fontsize=28)
#
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)#图框下边
# ax.spines['left'].set_linewidth(bwith)#图框左边
# ax.spines['top'].set_linewidth(bwith)#图框上边
# ax.spines['right'].set_linewidth(bwith)#图框右边
# # 保存并显示图像
#
# plt.savefig('E:/python_code/magnetostriction/SHAP/shap_figure/V.png', bbox_inches='tight', dpi=600)
# plt.show()




# plt.figure(figsize=(10, 8), dpi=600)
# plt.scatter(data['μ_ave.'], shap_values_data['μ_ave.'], s=10)

# plt.axhline(y=0, color='black', linestyle='-.', linewidth=1)
# plt.xlabel('μ_ave.', fontsize=12)
# plt.ylabel('SHAP value for μ_ave.', fontsize=12)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # plt.savefig("SHAP Dependence Plot_1.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# def plot_shap_dependence(feature_list, df, shap_values_df, file_name="SHAP_Dependence_Plots.pdf"):
#     fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=600)
#     plt.subplots_adjust(hspace=0.4, wspace=0.4)

#     for i, feature in enumerate(feature_list):
#         row = i // 3  # 行号
#         col = i % 3  # 列号
#         ax = axs[row, col]

#         ax.scatter(df[feature], shap_values_df[feature], s=10, alpha=0.7)

#         sns.regplot(x=df[feature], y=shap_values_df[feature], scatter=False, lowess=True, color='lightcoral', ax=ax)

#         lowess_data = sns.regplot(x=df[feature], y=shap_values_df[feature], scatter=False, lowess=True,
#                                   color='lightcoral', ax=ax)
#         line = lowess_data.get_lines()[0]
#         x_fit = line.get_xdata()
#         y_fit = line.get_ydata()

#         def find_zero_crossings(x_fit, y_fit):
#             crossings = []
#             for i in range(1, len(y_fit)):
#                 if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
#                     crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[i])[0]
#                     crossings.append(crossing)
#             return crossings
#
#         x_intercepts = find_zero_crossings(x_fit, y_fit)

#         for x_intercept in x_intercepts:
#             ax.axvline(x=x_intercept, color='blue', linestyle='--')
#             ax.text(x_intercept, 0.1, f'{x_intercept:.2f}', color='black', fontsize=10,
#                     verticalalignment='bottom')

#         ax.axhline(y=0, color='black', linestyle='-.', linewidth=1)

#         ax.set_xlabel(feature, fontsize=12)
#         ax.set_ylabel(f'SHAP value for\n{feature}', fontsize=12)

#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     axs[1, 2].axis('off')

#     # plt.savefig(file_name, format='pdf', bbox_inches='tight')
#     plt.show()
