import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
import shap
import matplotlib.pyplot as plt
import pandas as pd

output_dir = 'shap_plots'
os.makedirs(output_dir, exist_ok=True)

pgs_data = pd.read_csv('PGS.csv')
metabolites_data = pd.read_csv('metabolites.csv')

common_ids = pd.merge(pgs_data[['ID']], metabolites_data[['ID']], left_on='ID', right_on='ID')

pgs_filtered = pgs_data[pgs_data['ID'].isin(common_ids['ID'])]
metabolites_filtered = metabolites_data[metabolites_data['ID'].isin(common_ids['ID'])]

combined_df = pd.merge(pgs_filtered, metabolites_filtered, on='ID')

target_cols = ['a1', 'a2', 'a3', 'a4', 'a5']

def process_target(target_col, combined_df, output_dir):
    print(f"Processing target: {target_col}")
    
    non_feature_columns = ['ID', 'some_other_non_feature_column']
    X = combined_df.drop(columns=non_feature_columns)
    
    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
    
    X = X.select_dtypes(include=['float64', 'int64'])
    
    y = combined_df[target_col]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=1, step=1)
    
    rfe.fit(X, y)
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_
    }).sort_values(by='Ranking')
    
    print(f"Top features for {target_col}:")
    print(feature_ranking.head(15))
    
    k = 15
    top_features = feature_ranking[feature_ranking['Ranking'] <= k]['Feature']
    X_top_k = X[top_features]
    
    cross_val_scores = cross_val_score(rf, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_cv_mse = -cross_val_scores.mean()
    print(f'Mean CV MSE with top {k} features for {target_col}: {mean_cv_mse}')
    
    rf.fit(X_top_k, y)
    
    predictions = rf.predict(X_top_k)
    predictions_df = pd.DataFrame({
        'ID': combined_df['ID'],
        f'Actual_{target_col}': y,
        f'Predicted_{target_col}': predictions
    })
    predictions_df.to_csv(f'{output_dir}/predictions_{target_col}.csv', index=False)
    
    dummy = DummyRegressor(strategy="mean")
    dummy_scores = cross_val_score(dummy, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_dummy_mse = -dummy_scores.mean()
    print(f"Mean CV MSE of Dummy Regressor for {target_col}: {mean_dummy_mse}")
    
    explainer = shap.Explainer(rf, X_top_k)
    shap_values = explainer(X_top_k)
    
    plt.figure(figsize=(20, 12))
    shap.summary_plot(shap_values, X_top_k, show=False, plot_size=(20, 12))
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary_plot_{target_col}.pdf')
    plt.clf()
    
    print(f"Finished processing {target_col}\n")

for target_col in target_cols:
    process_target(target_col, combined_df, output_dir)
