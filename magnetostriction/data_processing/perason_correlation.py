import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('TbDyFe database.xlsx')
# print(data)

dataset = pd.DataFrame(data, columns=['r_emp.2','Tm_ave.','Tm_emp.1','Tb_ave.','Tb_emp.1','Hf_ave.','Hf_emp.1','Hv_ave.',
                                      'Hv_emp.1','Hs_ave.','Hs_emp.1','k_ave.','k_emp.1','μ_ave.','μ_emp.1','Vm_ave.',
                                      'Vm_emp.1','Mb_ave.','Mb_emp.2','Ms_ave.','Ms_emp.2','My_ave.','My_emp.2','En_ave.',
                                      'En_emp.1','VEC_ave.','VEC_emp.1','Ea_ave.','Ea_emp.2','Ei_ave.','Ei_emp.2','T_heat',
                                      't_heat','H_cri.'])
# print(dataset)
# dataset.info()
# print(dataset.describe())

corr_matrix = dataset.corr()

mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.tril_indices_from(mask, k=-1)] = True

plt.figure(figsize=(14, 11))
sns.heatmap(corr_matrix, cbar={"aspect":18}, mask=mask, annot=True, cmap='rainbow', fmt=".2f", vmin=-1, vmax=1,
            annot_kws={"size":6, "fontweight":"bold"})
# sns.heatmap(cm,
#                  cbar={"aspect":13},
#                  annot=True,
#                  square=True,
#                  cmap="rainbow",
#                  fmt='.2f',
#                  vmin=-1, vmax=1,
#                  annot_kws={'size':4, "fontweight":"bold"},
#                  yticklabels=attributes,
#                  xticklabels=attributes)
plt.savefig('E:/python_code/magnetostriction/data_processing/figure/pearson-1.png', bbox_inches='tight', dpi=600) # transparent=True,背景透明
plt.show()

# ////////////////MI Value////////////////////
#
from sklearn.feature_selection import mutual_info_regression

X = pd.DataFrame(data, columns=['Hs_ave.','Hs_emp.1', 'μ_ave.','μ_emp.1', 'Ms_ave.','Ms_emp.2','My_ave.','My_emp.2',
                                       'En_emp.1','VEC_ave.'])
y = pd.DataFrame(data, columns=['λa'])

mi_scores = mutual_info_regression(X, y)

plt.figure(figsize=(10, 6))
indices = np.arange(X.shape[1])
plt.bar(indices, mi_scores, color='skyblue', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Mutual Information Scores for Features')
plt.xticks(indices, rotation='vertical')
plt.tight_layout()
plt.show()

N = 10
selected_features = np.argsort(mi_scores)[::-1][:N]
print("Selected feature indices:", selected_features)

for i in range(N):
    print(f"Feature {selected_features[i]}: MI = {mi_scores[selected_features[i]]}")

