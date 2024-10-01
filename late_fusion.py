import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
import shap
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('shap_plots_CV_PRS', exist_ok=True)
os.makedirs('shap_plots_CV_metabolomics', exist_ok=True)

target_cols = ['a1', 'a2', 'a3', 'a4', 'a5']

for target_col in target_cols:
    print(f"Processing target: {target_col} for PGS data")

    X = df1_filtered.drop(columns=['ID1', 'a1', 'a2', 'a3', 'a4', 'a5', 'a'])

    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])

    y = df1_filtered[target_col]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=1, step=1)

    rfe.fit(X, y)
    ranking = rfe.ranking_
    features = X.columns

    feature_ranking = pd.DataFrame({'Feature': features, 'Ranking': ranking}).sort_values(by='Ranking')
    print(f"Feature ranking for {target_col}:")
    print(feature_ranking)

    k = 15
    top_features = feature_ranking[feature_ranking['Ranking'] <= k]['Feature']
    X_top_k = X[top_features]

    cross_val_scores = cross_val_score(rf, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_cross_val_score = -cross_val_scores.mean()
    print(f'Mean Cross-Validated MSE with top {k} features for {target_col}: {mean_cross_val_score}')

    rf.fit(X_top_k, y)

    explainer = shap.Explainer(rf, X_top_k)
    shap_values = explainer(X_top_k)

    plt.figure(figsize=(20, 12))
    shap.summary_plot(shap_values, X_top_k, show=False, plot_size=(20, 12))
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'shap_plots_CV_PRS/shap_summary_plot_df1_{target_col}.pdf')
    plt.clf()

    predictions = rf.predict(X_top_k)
    predictions_df = pd.DataFrame({
        'barcode': df1_filtered['ID1'],  
        f'Actual_{target_col}': y,
        f'Predicted_{target_col}': predictions
    })

    predictions_df.to_csv(f'shap_plots_CV_PRS/predictions_{target_col}.csv', index=False)

    dummy = DummyRegressor(strategy="mean")
    dummy_scores = cross_val_score(dummy, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_dummy_mse = -dummy_scores.mean()
    print(f"Mean Cross-Validated MSE of Dummy Regressor for {target_col}: {mean_dummy_mse}")

    print(f"Finished processing {target_col} for PGS data\n")

for target_col in target_cols:
    print(f"Processing target: {target_col} for metabolomics")

    X = df2_filtered.drop(columns=['barcode', 'a1', 'a2', 'a3', 'a4', 'a5', 'a'])

    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])

    y = df2_filtered[target_col]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=1, step=1)

    rfe.fit(X, y)
    ranking = rfe.ranking_
    features = X.columns

    feature_ranking = pd.DataFrame({'Feature': features, 'Ranking': ranking}).sort_values(by='Ranking')
    print(f"Feature ranking for {target_col}:")
    print(feature_ranking)

    k = 15
    top_features = feature_ranking[feature_ranking['Ranking'] <= k]['Feature']
    X_top_k = X[top_features]

    cross_val_scores = cross_val_score(rf, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_cross_val_score = -cross_val_scores.mean()
    print(f'Mean Cross-Validated MSE with top {k} features for {target_col} (metabolomics): {mean_cross_val_score}')

    rf.fit(X_top_k, y)

    explainer = shap.Explainer(rf, X_top_k)
    shap_values = explainer(X_top_k)

    plt.figure(figsize=(20, 12))
    shap.summary_plot(shap_values, X_top_k, show=False, plot_size=(20, 12))
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'shap_plots_CV_metabolomics/shap_summary_plot_df2_{target_col}.pdf')
    plt.clf()

    predictions = rf.predict(X_top_k)
    predictions_df = pd.DataFrame({
        'barcode': df2_filtered['barcode'],  
        f'Actual_{target_col}': y,
        f'Predicted_{target_col}': predictions
    })

    predictions_df.to_csv(f'shap_plots_CV_metabolomics/predictions_{target_col}.csv', index=False)

    dummy = DummyRegressor(strategy="mean")
    dummy_scores = cross_val_score(dummy, X_top_k, y, cv=5, scoring='neg_mean_squared_error')
    mean_dummy_mse = -dummy_scores.mean()
    print(f"Mean Cross-Validated MSE of Dummy Regressor for {target_col} (metabolomics): {mean_dummy_mse}")

    print(f"Finished processing {target_col} for metabolomics\n")

pred_a1_df1 = pd.read_csv('shap_plots_CV_PRS/predictions_a1.csv')
pred_a1_df2 = pd.read_csv('shap_plots_CV_metabolomics/predictions_a1.csv')

merged_a1 = pd.merge(pred_a1_df1, pred_a1_df2, left_on='barcode', right_on='barcode', suffixes=('_df1', '_df2'))

pred_a2_df1 = pd.read_csv('shap_plots_CV_PRS/predictions_a2.csv')
pred_a2_df2 = pd.read_csv('shap_plots_CV_metabolomics/predictions_a2.csv')
merged_a2 = pd.merge(pred_a2_df1, pred_a2_df2, left_on='barcode', right_on='barcode', suffixes=('_df1', '_df2'))

pred_a3_df1 = pd.read_csv('shap_plots_CV_PRS/predictions_a3.csv')
pred_a3_df2 = pd.read_csv('shap_plots_CV_metabolomics/predictions_a3.csv')
merged_a3 = pd.merge(pred_a3_df1, pred_a3_df2, left_on='barcode', right_on='barcode', suffixes=('_df1', '_df2'))

pred_a4_df1 = pd.read_csv('shap_plots_CV_PRS/predictions_a4.csv')
pred_a4_df2 = pd.read_csv('shap_plots_CV_metabolomics/predictions_a4.csv')
merged_a4 = pd.merge(pred_a4_df1, pred_a4_df2, left_on='barcode', right_on='barcode', suffixes=('_df1', '_df2'))

pred_a5_df1 = pd.read_csv('shap_plots_CV_PRS/predictions_a5.csv')
pred_a5_df2 = pd.read_csv('shap_plots_CV_metabolomics/predictions_a5.csv')
merged_a5 = pd.merge(pred_a5_df1, pred_a5_df2, left_on='barcode', right_on='barcode', suffixes=('_df1', '_df2'))

merged_predictions = merged_a1.merge(merged_a2, on=['barcode', 'barcode'])
merged_predictions = merged_predictions.merge(merged_a3, on=['barcode', 'barcode'])
merged_predictions = merged_predictions.merge(merged_a4, on=['barcode', 'barcode'])
merged_predictions = merged_predictions.merge(merged_a5, on=['barcode', 'barcode'])

merged_predictions['Final_Pred_a1'] = (merged_predictions['Predicted_a1_df1'] + merged_predictions['Predicted_a1_df2']) / 2
merged_predictions['Final_Pred_a2'] = (merged_predictions['Predicted_a2_df1'] + merged_predictions['Predicted_a2_df2']) / 2
merged_predictions['Final_Pred_a3'] = (merged_predictions['Predicted_a3_df1'] + merged_predictions['Predicted_a3_df2']) / 2
merged_predictions['Final_Pred_a4'] = (merged_predictions['Predicted_a4_df1'] + merged_predictions['Predicted_a4_df2']) / 2
merged_predictions['Final_Pred_a5'] = (merged_predictions['Predicted_a5_df1'] + merged_predictions['Predicted_a5_df2']) / 2

merged_predictions.to_csv('shap_plots_CV_metabolomics/merged_late_fusion_predictions.csv', index=False)

print("Late fusion predictions have been saved successfully.")
