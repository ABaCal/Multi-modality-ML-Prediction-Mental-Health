import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_filtered = merged_df.drop(columns=['sona_id', 'a'])

labels = ['a1', 'a2', 'a3', 'a4', 'a5']

results_list = []

scaler = StandardScaler()
data_scaled = data_filtered.copy()
columns_to_normalize = [col for col in data_filtered.columns if col not in labels + ['age', 'gender']]
data_scaled[columns_to_normalize] = scaler.fit_transform(data_filtered[columns_to_normalize])

for variable in data_scaled.columns:
    if variable not in labels + ['age', 'gender']: 
        for label in labels:
            X = data_scaled[[variable, 'age', 'gender']]
            y = data_scaled[label]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()
            coef = model.params.iloc[1]  
            p_value = model.pvalues.iloc[1]  

            results_list.append({
                'variable': variable,
                'label': label,
                'coef': coef,
                'p_value': p_value
            })

results_df = pd.DataFrame(results_list)

coef_pivot = results_df.pivot(index='variable', columns='label', values='coef')
pval_pivot = results_df.pivot(index='variable', columns='label', values='p_value')

significant_rows = pval_pivot[(pval_pivot < 0.05).any(axis=1)]
coef_pivot_significant = coef_pivot.loc[significant_rows.index]
pval_pivot_significant = pval_pivot.loc[significant_rows.index]

annotations = pval_pivot_significant.applymap(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')

num_rows = coef_pivot_significant.shape[0]
num_cols = coef_pivot_significant.shape[1]
cmap = 'bwr'

cluster_grid = sns.clustermap(
    coef_pivot_significant,
    cmap=cmap,
    center=0,
    col_cluster=False,
    figsize=(num_cols * 2, num_rows * 0.5),
    dendrogram_ratio=(0.1, 0.2),
    cbar_pos=(0.8, 0.8, 0.03, 0.18),
    annot=annotations,
    fmt='',
    annot_kws={"size": 12, "color": "white"}
)

cbar = cluster_grid.ax_heatmap.collections[0].colorbar
cbar.set_ticks([coef_pivot_significant.min().min(), 0, coef_pivot_significant.max().max()])
cbar.set_ticklabels([f'{coef_pivot_significant.min().min():.2f}', '0', f'{coef_pivot_significant.max().max():.2f}'])

cluster_grid.fig.suptitle('Clustered Heatmap of Normalized Regression Coefficients Proteomics', y=1.05)
plt.show()
