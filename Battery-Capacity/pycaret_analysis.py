#!/usr/bin/env python3
"""
PyCaret AutoML Analysis for Battery Capacity Prediction

This script uses PyCaret's regression module to automatically test and compare
multiple machine learning models on the battery capacity dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyCaret imports
from pycaret.regression import *

# Custom modules
from src.data_loader import DataLoader

def main():
    """Main function to run PyCaret analysis"""
    
    print("=" * 80)
    print("PYCARET AUTOML ANALYSIS FOR BATTERY CAPACITY PREDICTION")
    print("=" * 80)
    
    # Initialize paths
    current_dir = Path.cwd()
    data_path = current_dir / "dataset_299.xlsx"
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Data path: {data_path}")
    print(f"📁 Results directory: {results_dir}")
    print(f"✅ Data file exists: {data_path.exists()}")
    
    # Load and prepare data
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*50)
    
    data_loader = DataLoader(data_path)
    df = data_loader.load_data()
    X, y = data_loader.split_features_target(df)
    
    # Create a combined dataset for PyCaret (it expects target column in the dataframe)
    data = X.copy()
    data['target'] = y
    
    print(f"📊 Dataset shape: {data.shape}")
    print(f"🎯 Target variable: 'target' (Average Capacity)")
    print(f"🔢 Number of features: {X.shape[1]}")
    print(f"📈 Target statistics:")
    print(y.describe())
    
    # Setup PyCaret environment
    print("\n" + "="*50)
    print("STEP 2: PYCARET ENVIRONMENT SETUP")
    print("="*50)
    
    # Setup regression environment
    reg = setup(
        data=data,
        target='target',
        session_id=123,
        train_size=0.8,
        fold=5,  # 5-fold cross-validation
        verbose=False,  # Replaced 'silent=True' with 'verbose=False' for PyCaret 3.3.2
        use_gpu=False  # Set to True if you have GPU support
    )
    
    print("✅ PyCaret environment successfully set up!")
    print(f"📊 Training set size: {int(0.8 * len(data))} samples")
    print(f"📊 Test set size: {int(0.2 * len(data))} samples")
    print("🔄 Cross-validation: 5-fold")
    
    # Compare models
    print("\n" + "="*50)
    print("STEP 3: MODEL COMPARISON")
    print("="*50)
    print("🤖 Training and comparing multiple regression models...")
    print("This may take a few minutes...")
    
    # Compare all available models
    model_comparison = compare_models(
        include=[
            'lr',      # Linear Regression
            'lasso',   # Lasso Regression
            'ridge',   # Ridge Regression
            'en',      # Elastic Net
            'huber',   # Huber Regressor
            'rf',      # Random Forest
            'et',      # Extra Trees
            'gbr',     # Gradient Boosting
            'lightgbm', # LightGBM
            'xgboost', # XGBoost
            'catboost', # CatBoost
            'knn',     # K-Nearest Neighbors
            'mlp',     # Multi-Layer Perceptron
            'svm',     # Support Vector Machine
            'dt',      # Decision Tree
            'ada',     # AdaBoost
            'br'       # Bayesian Ridge
        ],
        sort='RMSE',  # Sort by RMSE (lower is better)
        n_select=10,  # Select top 10 models for detailed analysis
        verbose=False
    )
    
    print("✅ Model comparison completed!")
    print("\n📊 TOP 10 MODELS COMPARISON:")
    print("="*80)
    print(model_comparison.round(4))
    
    # Save comparison results
    comparison_file = results_dir / "pycaret_model_comparison.csv"
    model_comparison.to_csv(comparison_file)
    print(f"\n💾 Results saved to: {comparison_file}")
    
    # Get the best model
    best_models = compare_models(
        include=[
            'lr', 'lasso', 'ridge', 'en', 'huber', 'rf', 'et', 'gbr',
            'lightgbm', 'xgboost', 'catboost', 'knn', 'mlp', 'svm', 'dt', 'ada', 'br'
        ],
        sort='RMSE',
        n_select=1,
        verbose=False
    )
    
    best_model = best_models[0] if isinstance(best_models, list) else best_models
    
    print("\n" + "="*50)
    print("STEP 4: BEST MODEL ANALYSIS")
    print("="*50)
    
    # Create and evaluate the best model
    print("🏆 Training the best performing model...")
    created_model = create_model(best_model, verbose=False)
    
    # Tune hyperparameters
    print("🔧 Tuning hyperparameters...")
    tuned_model = tune_model(created_model, verbose=False)
    
    # Evaluate the tuned model
    print("📊 Evaluating the tuned model...")
    evaluate_model(tuned_model)
    
    # Finalize the model (trains on full dataset)
    print("🎯 Finalizing model on full dataset...")
    final_model = finalize_model(tuned_model)
    
    # Make predictions on test set
    print("🔮 Making predictions on test set...")
    predictions = predict_model(final_model)
    
    print("\n📈 PREDICTION RESULTS (First 10 samples):")
    print("="*50)
    print(predictions[['target', 'prediction_label', 'prediction_residuals']].head(10).round(4))
    
    # Calculate additional metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    y_true = predictions['target']
    y_pred = predictions['prediction_label']
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    print("\n📊 FINAL MODEL PERFORMANCE:")
    print("="*40)
    print(f"🎯 Mean Absolute Error (MAE): {mae:.6f}")
    print(f"📏 Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"📈 R² Score: {r2:.6f}")
    print(f"📊 Model Type: {type(final_model).__name__}")
    
    # Save the final model
    model_file = results_dir / "pycaret_best_model.pkl"
    save_model(final_model, str(model_file))
    print(f"\n💾 Best model saved to: {model_file}.pkl")
    
    # Save predictions
    predictions_file = results_dir / "pycaret_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print(f"💾 Predictions saved to: {predictions_file}")
    
    # Feature importance (if available)
    try:
        print("\n" + "="*50)
        print("STEP 5: FEATURE IMPORTANCE")
        print("="*50)
        
        # Plot feature importance
        print("📊 Generating feature importance plot...")
        plot_model(final_model, plot='feature', save=True)
        print("💾 Feature importance plot saved as 'Feature Importance.png'")
        
        # Try to get feature importance values
        if hasattr(final_model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
            print("="*40)
            print(importance_df.head(10).round(6))
            
            # Save feature importance
            importance_file = results_dir / "pycaret_feature_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            print(f"\n💾 Feature importance saved to: {importance_file}")
            
    except Exception as e:
        print(f"⚠️  Feature importance not available for this model type: {e}")
    
    # Generate additional plots
    print("\n" + "="*50)
    print("STEP 6: GENERATING VISUALIZATION PLOTS")
    print("="*50)
    
    try:
        print("📊 Generating residuals plot...")
        plot_model(final_model, plot='residuals', save=True)
        
        print("📊 Generating prediction error plot...")
        plot_model(final_model, plot='error', save=True)
        
        print("📊 Generating learning curve...")
        plot_model(final_model, plot='learning', save=True)
        
        print("📊 Generating validation curve...")
        plot_model(final_model, plot='vc', save=True)
        
        print("✅ All plots generated and saved!")
        
    except Exception as e:
        print(f"⚠️  Some plots could not be generated: {e}")
    
    print("\n" + "="*80)
    print("🎉 PYCARET ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Best Model: {type(final_model).__name__}")
    print(f"🎯 Final RMSE: {rmse:.6f}")
    print(f"📈 Final R² Score: {r2:.6f}")
    print(f"📁 All results saved in: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
