# ========================================
# COMPREHENSIVE FEATURE SELECTION ANALYSIS
# Add this cell to your TPOT notebook
# ========================================

print("\n" + "=" * 80)
print("🔍 COMPREHENSIVE FEATURE SELECTION ANALYSIS")
print("=" * 80)
print("Comparing different feature selection methods on battery capacity data")

# Import enhanced preprocessor
from src.preprocessor import Preprocessor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Define feature selection methods to test
feature_selection_methods = {
    'No Selection': None,
    'Pearson Correlation': 'pearson',
    'Mutual Information': 'mutual_info',
    'F-Regression (ANOVA)': 'f_regression'
}

# Test different numbers of features
n_features_to_test = [10, 20, 30, 40, 50]  # Adjust based on your dataset

print(f"🧮 Original dataset: {X.shape[1]} features, {X.shape[0]} samples")
print(f"🔬 Testing {len(feature_selection_methods)} feature selection methods")
print(f"📊 Testing feature counts: {n_features_to_test}")

# Store results
feature_selection_results = []

# Test each feature selection method
for method_name, method_code in feature_selection_methods.items():
    print(f"\n{'='*60}")
    print(f"📊 TESTING: {method_name}")
    print(f"{'='*60}")
    
    if method_code is None:
        # No feature selection - use all features
        print("Using all features (no selection)")
        
        # Simple preprocessing without feature selection
        preprocessor = Preprocessor(scaler_type='standard', feature_selection=None)
        X_processed, y_processed = preprocessor.preprocess_data(X_train, y_train, fit=True)
        
        # Test with Ridge regression
        ridge_model = Ridge(alpha=1.0, random_state=123)
        cv_scores = cross_val_score(ridge_model, X_processed, y_processed, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        feature_selection_results.append({
            'Method': method_name,
            'N_Features': X.shape[1],
            'CV_RMSE': cv_rmse,
            'CV_STD': np.sqrt(-cv_scores).std(),
            'Selected_Features': None
        })
        
        print(f"   Features used: {X.shape[1]} (all)")
        print(f"   CV RMSE: {cv_rmse:.6f} (±{np.sqrt(-cv_scores).std():.6f})")
        
    else:
        # Test different numbers of features
        for n_features in n_features_to_test:
            if n_features >= X.shape[1]:
                continue  # Skip if requesting more features than available
                
            print(f"\n🔍 Testing {method_name} with {n_features} features...")
            
            # Create preprocessor with feature selection
            preprocessor = Preprocessor(
                scaler_type='standard', 
                feature_selection=method_code,
                n_features=n_features
            )
            
            try:
                # Fit preprocessor and get selected features
                X_processed, y_processed = preprocessor.preprocess_data(X_train, y_train, fit=True)
                
                # Get feature importance scores
                feature_scores = preprocessor.get_feature_importance()
                selected_feature_names = preprocessor.get_selected_feature_names()
                
                # Test with Ridge regression
                ridge_model = Ridge(alpha=1.0, random_state=123)
                cv_scores = cross_val_score(ridge_model, X_processed, y_processed, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                feature_selection_results.append({
                    'Method': method_name,
                    'N_Features': n_features,
                    'CV_RMSE': cv_rmse,
                    'CV_STD': np.sqrt(-cv_scores).std(),
                    'Selected_Features': selected_feature_names,
                    'Feature_Scores': feature_scores
                })
                
                print(f"   Features selected: {len(selected_feature_names)}")
                print(f"   CV RMSE: {cv_rmse:.6f} (±{np.sqrt(-cv_scores).std():.6f})")
                
                # Display top 5 selected features
                if feature_scores is not None and len(feature_scores) > 0:
                    print(f"   Top 5 features:")
                    top_features = feature_scores.head(5)
                    for idx, row in top_features.iterrows():
                        if method_code == 'pearson':
                            print(f"     • {row['feature']}: correlation = {row['abs_correlation']:.4f}")
                        elif method_code == 'mutual_info':
                            print(f"     • {row['feature']}: MI score = {row['mutual_info_score']:.4f}")
                        elif method_code == 'f_regression':
                            print(f"     • {row['feature']}: F-score = {row['f_score']:.4f}")
                            
            except Exception as e:
                print(f"   ❌ Error with {method_name} ({n_features} features): {e}")

# Create results DataFrame
results_df = pd.DataFrame(feature_selection_results)

print(f"\n{'='*80}")
print("📊 FEATURE SELECTION RESULTS SUMMARY")
print(f"{'='*80}")

# Display results table
display(results_df[['Method', 'N_Features', 'CV_RMSE', 'CV_STD']].round(6))

# Find best performing method for each feature count
print(f"\n🏆 BEST METHODS BY FEATURE COUNT:")
print("-" * 50)

for n_feat in [None] + n_features_to_test:
    if n_feat is None:
        subset = results_df[results_df['Method'] == 'No Selection']
        feat_label = "All Features"
    else:
        subset = results_df[results_df['N_Features'] == n_feat]
        feat_label = f"{n_feat} Features"
    
    if len(subset) > 0:
        best_method = subset.loc[subset['CV_RMSE'].idxmin()]
        print(f"   {feat_label}: {best_method['Method']} (RMSE: {best_method['CV_RMSE']:.6f})")

# Find overall best method
best_overall = results_df.loc[results_df['CV_RMSE'].idxmin()]
print(f"\n🥇 OVERALL BEST: {best_overall['Method']} with {best_overall['N_Features']} features")
print(f"   CV RMSE: {best_overall['CV_RMSE']:.6f} (±{best_overall['CV_STD']:.6f})")

# Save detailed results
feature_selection_file = tpot_dir / 'feature_selection_comparison.csv'
results_df[['Method', 'N_Features', 'CV_RMSE', 'CV_STD']].to_csv(feature_selection_file, index=False)
print(f"\n💾 Feature selection results saved to: {feature_selection_file}")

# Visualization
print(f"\n📈 Creating feature selection comparison plots...")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: RMSE comparison by method and feature count
plt.subplot(2, 2, 1)
methods_with_features = results_df[results_df['Method'] != 'No Selection']
if len(methods_with_features) > 0:
    for method in methods_with_features['Method'].unique():
        method_data = methods_with_features[methods_with_features['Method'] == method]
        plt.plot(method_data['N_Features'], method_data['CV_RMSE'], 'o-', label=method, linewidth=2, markersize=6)
    
    # Add horizontal line for no selection baseline
    no_selection_rmse = results_df[results_df['Method'] == 'No Selection']['CV_RMSE'].iloc[0]
    plt.axhline(y=no_selection_rmse, color='red', linestyle='--', alpha=0.7, label='No Selection (All Features)')
    
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Cross-Validation RMSE')
    plt.title('Feature Selection Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 2: Bar chart of best RMSE for each method
plt.subplot(2, 2, 2)
method_best_rmse = results_df.groupby('Method')['CV_RMSE'].min().sort_values()
colors = ['green' if x == method_best_rmse.min() else 'skyblue' for x in method_best_rmse.values]
bars = plt.bar(range(len(method_best_rmse)), method_best_rmse.values, color=colors)
plt.xticks(range(len(method_best_rmse)), method_best_rmse.index, rotation=45, ha='right')
plt.ylabel('Best CV RMSE')
plt.title('Best Performance by Feature Selection Method')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{height:.4f}', 
             ha='center', va='bottom', fontsize=9)

# Plot 3: Feature importance visualization for best method
plt.subplot(2, 2, 3)
if 'Feature_Scores' in best_overall and best_overall['Feature_Scores'] is not None:
    feature_scores = best_overall['Feature_Scores']
    if len(feature_scores) > 0:
        top_10_features = feature_scores.head(10)
        
        if best_overall['Method'] == 'Pearson Correlation':
            y_values = top_10_features['abs_correlation']
            y_label = 'Absolute Correlation'
        elif best_overall['Method'] == 'Mutual Information':
            y_values = top_10_features['mutual_info_score']
            y_label = 'Mutual Information Score'
        elif best_overall['Method'] == 'F-Regression (ANOVA)':
            y_values = top_10_features['f_score']
            y_label = 'F-Score'
        
        plt.barh(range(len(top_10_features)), y_values, color='orange', alpha=0.7)
        plt.yticks(range(len(top_10_features)), 
                  [f"{feat[:20]}..." if len(feat) > 20 else feat for feat in top_10_features['feature']])
        plt.xlabel(y_label)
        plt.title(f'Top 10 Features - {best_overall["Method"]}')
        plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No feature scores available\nfor best method', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Feature Importance Not Available')

# Plot 4: Performance improvement over baseline
plt.subplot(2, 2, 4)
baseline_rmse = results_df[results_df['Method'] == 'No Selection']['CV_RMSE'].iloc[0]
results_with_selection = results_df[results_df['Method'] != 'No Selection'].copy()
results_with_selection['Improvement_%'] = ((baseline_rmse - results_with_selection['CV_RMSE']) / baseline_rmse) * 100

if len(results_with_selection) > 0:
    best_improvements = results_with_selection.groupby('Method')['Improvement_%'].max().sort_values(ascending=False)
    colors = ['green' if x > 0 else 'red' for x in best_improvements.values]
    bars = plt.bar(range(len(best_improvements)), best_improvements.values, color=colors, alpha=0.7)
    plt.xticks(range(len(best_improvements)), best_improvements.index, rotation=45, ha='right')
    plt.ylabel('Best Improvement over Baseline (%)')
    plt.title('Performance Improvement vs No Selection')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.3, 
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()

# Save the plot
feature_selection_plot = tpot_dir / 'feature_selection_analysis.png'
plt.savefig(feature_selection_plot, dpi=300, bbox_inches='tight')
plt.show()

print(f"📊 Feature selection analysis plot saved to: {feature_selection_plot}")

print(f"\n{'='*80}")
print("✅ FEATURE SELECTION ANALYSIS COMPLETED!")
print(f"{'='*80}")
print(f"🏆 Best method: {best_overall['Method']} with {best_overall['N_Features']} features")
print(f"📊 Best CV RMSE: {best_overall['CV_RMSE']:.6f}")
if len(results_with_selection) > 0:
    best_improvement = results_with_selection['Improvement_%'].max()
    print(f"📈 Best improvement: {best_improvement:.2f}% over baseline")
print(f"💾 Results saved to: {feature_selection_file}")
print(f"📊 Visualization saved to: {feature_selection_plot}")
