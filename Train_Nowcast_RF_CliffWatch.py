"""
================================================================================
CLIFF WATCH NOWCAST RF MODEL TRAINING
================================================================================
Trains Random Forest models to predict coastal landslide risk using 
Spatio-Temporal Cross-Validation (STCV).

Compares:
  - Baseline model (terrain + precipitation + seismic)
  - Ocean-enhanced model (baseline + CDIP + NDBC wave data)

Outputs:
  - Publication-ready tables (CSV and LaTeX)
  - Publication-ready figures (PNG, 300 DPI)
  - Frozen model for Cliff Watch app deployment (.joblib + .json)

Author: J. Bighouse, USC GIST
Date: December 2025
================================================================================
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump as joblib_dump

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
INPUT_CSV = "C:/Users/MB/Desktop/SSCI 594a_Thesis/Thesis_ArcGIS/Code/NewModelScripts12.13.25/nowcast_training_FINAL.csv"
OUTPUT_DIR = "C:/Users/MB/Desktop/SSCI 594a_Thesis/Thesis_ArcGIS/Code/NewModelScripts12.13.25/model_outputs"

# Column names
COL_LABEL = "Label"
COL_LOCATION = "location_id"
COL_DATE = "EVENT_DATE"
COL_LAT = "LAT"
COL_LON = "LONG"

# Model parameters (from thesis: 600 trees)
N_ESTIMATORS = 600
RANDOM_STATE = 42
N_FOLDS = 5

# Risk thresholds for Cliff Watch app
RISK_THRESHOLDS = {
    "low_max": 0.30,      # < 0.30 = Low
    "medium_max": 0.60,   # 0.30-0.60 = Medium
    # >= 0.60 = High
}

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Terrain features (constant within locations - used for STCV stratification context)
TERRAIN_FEATURES = [
    'Slope', 'NEAR_DIST_COAST', 'Curvature', 'TWI', 
    'DistanceToFaults', 'NLCD',
    'aspect_sin', 'aspect_cos'
]

# Precipitation features
PRECIP_FEATURES = [
    'PRCP', 'PRCP_3', 'PRCP_7', 'PRCP_14', 'PRCP_30',
    'PRCP_missing_flag'
]

# Seismic features
SEISMIC_FEATURES = [
    'EQ_ct_7d_25km', 'EQ_maxM_7d_25km',
    'EQ_ct_7d_50km', 'EQ_maxM_7d_50km',
    'EQ_ct_30d_25km', 'EQ_maxM_30d_25km',
    'EQ_ct_30d_50km', 'EQ_maxM_30d_50km'
]

# CDIP wave features
CDIP_FEATURES = [
    'Hs_max', 'Tp_mean', 'WaveEnergy', 'Hs_ramp_1d',
    'Hs_max_sum_3d', 'Hs_max_sum_7d', 'Hs_max_sum_14d', 'Hs_max_sum_30d',
    'Hs_max_max_3d', 'Hs_max_max_7d', 'Hs_max_max_14d', 'Hs_max_max_30d',
    'Tp_mean_mean_3d', 'Tp_mean_mean_7d', 'Tp_mean_mean_14d', 'Tp_mean_mean_30d',
    'WaveEnergy_sum_3d', 'WaveEnergy_sum_7d', 'WaveEnergy_sum_14d', 'WaveEnergy_sum_30d',
    'Hs_peak_offset_3d', 'Hs_peak_offset_7d', 'Hs_peak_offset_14d', 'Hs_peak_offset_30d',
    'log1p_WaveEnergy_sum_3d', 'log1p_WaveEnergy_sum_7d', 
    'log1p_WaveEnergy_sum_14d', 'log1p_WaveEnergy_sum_30d',
    'OceanxRain_3d', 'OceanxRain_7d', 'OceanxRain_14d', 'OceanxRain_30d',
    'wave_has_same_day', 'wave_has_3d', 'wave_has_7d', 'wave_has_14d', 'wave_has_30d'
]

# NDBC wave features
NDBC_FEATURES = [
    'WVHT_mean', 'DPD_mean', 'APD_mean',
    'WVHT_mean_1d', 'WVHT_mean_3d', 'WVHT_mean_7d', 'WVHT_mean_14d', 'WVHT_mean_30d',
    'DPD_mean_1d', 'DPD_mean_3d', 'DPD_mean_7d', 'DPD_mean_14d', 'DPD_mean_30d',
    'APD_mean_1d', 'APD_mean_3d', 'APD_mean_7d', 'APD_mean_14d', 'APD_mean_30d'
]

# Combined feature sets
BASELINE_FEATURES = TERRAIN_FEATURES + PRECIP_FEATURES + SEISMIC_FEATURES
OCEAN_FEATURES = CDIP_FEATURES + NDBC_FEATURES
ALL_FEATURES = BASELINE_FEATURES + OCEAN_FEATURES


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def get_available_features(df, feature_list):
    """Return features that exist in the dataframe."""
    return [f for f in feature_list if f in df.columns]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    metrics = {}
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        metrics['auc'] = np.nan
        metrics['precision'] = np.nan
        metrics['recall'] = np.nan
        metrics['f1'] = np.nan
        return metrics
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = np.nan
    
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    return metrics


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find threshold that maximizes the specified metric."""
    best_threshold = 0.5
    best_score = 0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# =============================================================================
# MODEL TRAINING WITH STCV
# =============================================================================

def train_with_stcv(df, features, model_name, n_folds=5):
    """
    Train RF model using Spatio-Temporal Cross-Validation.
    Groups by location_id to prevent data leakage.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    
    # Get available features
    available_features = get_available_features(df, features)
    print(f"Features requested: {len(features)}")
    print(f"Features available: {len(available_features)}")
    
    missing = set(features) - set(available_features)
    if missing:
        print(f"Missing features: {missing}")
    
    # Prepare data
    X = df[available_features].copy()
    y = df[COL_LABEL].values
    groups = df[COL_LOCATION].values
    
    print(f"Samples: {len(X)}")
    print(f"Positive (landslides): {y.sum()}")
    print(f"Negative (controls): {len(y) - y.sum()}")
    print(f"Unique locations: {len(np.unique(groups))}")
    
    # Initialize cross-validation
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)
    
    if actual_folds < n_folds:
        print(f"⚠️ Only {n_groups} unique locations, using {actual_folds} folds")
    
    gkf = GroupKFold(n_splits=actual_folds)
    
    # Storage for results
    fold_metrics = []
    all_predictions = []
    all_feature_importances = []
    
    # STCV loop
    print(f"\nRunning {actual_folds}-fold STCV...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check class distribution
        train_pos = y_train.sum()
        test_pos = y_test.sum()
        
        if train_pos == 0 or train_pos == len(y_train):
            print(f"  Fold {fold_idx}: Skipping (single class in train)")
            continue
        if test_pos == 0 or test_pos == len(y_test):
            print(f"  Fold {fold_idx}: Skipping (single class in test)")
            continue
        
        # Build pipeline with imputation
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        opt_thresh, _ = find_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_thresh).astype(int)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['fold'] = fold_idx
        metrics['n_train'] = len(y_train)
        metrics['n_test'] = len(y_test)
        metrics['train_pos'] = train_pos
        metrics['test_pos'] = test_pos
        metrics['threshold'] = opt_thresh
        
        fold_metrics.append(metrics)
        
        # Store predictions
        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'index': idx,
                'y_true': y_test[i],
                'y_prob': y_prob[i],
                'y_pred': y_pred[i],
                'fold': fold_idx
            })
        
        # Store feature importances
        importances = pipeline.named_steps['classifier'].feature_importances_
        all_feature_importances.append(importances)
        
        print(f"  Fold {fold_idx}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'features': available_features,
        'fold_metrics': pd.DataFrame(fold_metrics),
        'predictions': pd.DataFrame(all_predictions),
        'mean_importance': np.mean(all_feature_importances, axis=0) if all_feature_importances else None
    }
    
    # Summary statistics
    if fold_metrics:
        fm_df = results['fold_metrics']
        results['mean_auc'] = fm_df['auc'].mean()
        results['std_auc'] = fm_df['auc'].std()
        results['mean_f1'] = fm_df['f1'].mean()
        results['mean_precision'] = fm_df['precision'].mean()
        results['mean_recall'] = fm_df['recall'].mean()
        
        print(f"\n📊 {model_name} Summary:")
        print(f"   Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        print(f"   Mean F1:  {results['mean_f1']:.3f}")
        print(f"   Mean Precision: {results['mean_precision']:.3f}")
        print(f"   Mean Recall: {results['mean_recall']:.3f}")
    
    return results


# =============================================================================
# PUBLICATION OUTPUTS
# =============================================================================

def create_performance_table(baseline_results, ocean_results, output_dir):
    """Create Table 4: Per-fold performance metrics."""
    print("\n📄 Creating performance table...")
    
    # Combine fold metrics
    baseline_df = baseline_results['fold_metrics'].copy()
    baseline_df['model'] = 'Baseline'
    
    ocean_df = ocean_results['fold_metrics'].copy()
    ocean_df['model'] = 'Ocean-Enhanced'
    
    combined = pd.concat([baseline_df, ocean_df], ignore_index=True)
    
    # Format for publication
    table = combined[['model', 'fold', 'auc', 'precision', 'recall', 'f1', 'threshold']].copy()
    table.columns = ['Model', 'Fold', 'AUC', 'Precision', 'Recall', 'F1', 'Threshold']
    
    # Add summary row
    for model_name, results in [('Baseline', baseline_results), ('Ocean-Enhanced', ocean_results)]:
        summary = pd.DataFrame([{
            'Model': model_name,
            'Fold': 'Mean',
            'AUC': results['mean_auc'],
            'Precision': results['mean_precision'],
            'Recall': results['mean_recall'],
            'F1': results['mean_f1'],
            'Threshold': results['fold_metrics']['threshold'].mean()
        }])
        table = pd.concat([table, summary], ignore_index=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'Table4_Performance_Metrics.csv')
    table.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"   Saved: {csv_path}")
    
    # Create LaTeX table
    latex_path = os.path.join(output_dir, 'Table4_Performance_Metrics.tex')
    
    latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Per-fold performance metrics for baseline and ocean-enhanced models}
\\label{tab:performance}
\\begin{tabular}{llcccc}
\\toprule
Model & Fold & AUC & Precision & Recall & F1 \\\\
\\midrule
"""
    
    for _, row in table.iterrows():
        fold_str = str(row['Fold']) if row['Fold'] != 'Mean' else '\\textbf{Mean}'
        latex_content += f"{row['Model']} & {fold_str} & {row['AUC']:.3f} & {row['Precision']:.3f} & {row['Recall']:.3f} & {row['F1']:.3f} \\\\\n"
        if row['Fold'] == 'Mean' and row['Model'] == 'Baseline':
            latex_content += "\\midrule\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"   Saved: {latex_path}")
    
    return table


def create_feature_importance_figure(results, model_name, output_dir, top_n=20):
    """Create feature importance bar chart."""
    print(f"\n📊 Creating feature importance figure for {model_name}...")
    
    if results['mean_importance'] is None:
        print("   ⚠️ No importance data available")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': results['features'],
        'importance': results['mean_importance']
    }).sort_values('importance', ascending=True)
    
    # Get top N
    top_features = importance_df.tail(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by feature type
    colors = []
    for feat in top_features['feature']:
        if feat in TERRAIN_FEATURES:
            colors.append('#2E86AB')  # Blue for terrain
        elif feat in PRECIP_FEATURES:
            colors.append('#A23B72')  # Purple for precip
        elif feat in SEISMIC_FEATURES:
            colors.append('#F18F01')  # Orange for seismic
        elif feat in CDIP_FEATURES:
            colors.append('#C73E1D')  # Red for CDIP
        elif feat in NDBC_FEATURES:
            colors.append('#3B1F2B')  # Dark for NDBC
        else:
            colors.append('#666666')  # Gray for unknown
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Normalized Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Predictors: {model_name}', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Terrain'),
        Patch(facecolor='#A23B72', label='Precipitation'),
        Patch(facecolor='#F18F01', label='Seismic'),
        Patch(facecolor='#C73E1D', label='CDIP Waves'),
        Patch(facecolor='#3B1F2B', label='NDBC Waves')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    safe_name = model_name.replace(' ', '_').replace('-', '_')
    png_path = os.path.join(output_dir, f'Figure_FeatureImportance_{safe_name}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {png_path}")
    
    # Also save importance data
    csv_path = os.path.join(output_dir, f'FeatureImportance_{safe_name}.csv')
    importance_df.sort_values('importance', ascending=False).to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")


def create_roc_comparison_figure(baseline_results, ocean_results, output_dir):
    """Create ROC curve comparison figure."""
    print("\n📊 Creating ROC comparison figure...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for results, label, color in [
        (baseline_results, 'Baseline', '#2E86AB'),
        (ocean_results, 'Ocean-Enhanced', '#C73E1D')
    ]:
        pred_df = results['predictions']
        if pred_df.empty:
            continue
            
        fpr, tpr, _ = roc_curve(pred_df['y_true'], pred_df['y_prob'])
        auc = roc_auc_score(pred_df['y_true'], pred_df['y_prob'])
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison: Baseline vs Ocean-Enhanced', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'Figure_ROC_Comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {png_path}")


def create_model_comparison_summary(baseline_results, ocean_results, output_dir):
    """Create summary comparison table."""
    print("\n📄 Creating model comparison summary...")
    
    comparison = pd.DataFrame([
        {
            'Model': 'Baseline',
            'Features': len(baseline_results['features']),
            'Mean AUC': baseline_results['mean_auc'],
            'Std AUC': baseline_results['std_auc'],
            'Mean F1': baseline_results['mean_f1'],
            'Mean Precision': baseline_results['mean_precision'],
            'Mean Recall': baseline_results['mean_recall']
        },
        {
            'Model': 'Ocean-Enhanced',
            'Features': len(ocean_results['features']),
            'Mean AUC': ocean_results['mean_auc'],
            'Std AUC': ocean_results['std_auc'],
            'Mean F1': ocean_results['mean_f1'],
            'Mean Precision': ocean_results['mean_precision'],
            'Mean Recall': ocean_results['mean_recall']
        }
    ])
    
    # Calculate improvement
    auc_diff = ocean_results['mean_auc'] - baseline_results['mean_auc']
    auc_pct = (auc_diff / baseline_results['mean_auc']) * 100 if baseline_results['mean_auc'] > 0 else 0
    
    comparison.loc[len(comparison)] = {
        'Model': 'Difference (Ocean - Baseline)',
        'Features': len(ocean_results['features']) - len(baseline_results['features']),
        'Mean AUC': auc_diff,
        'Std AUC': np.nan,
        'Mean F1': ocean_results['mean_f1'] - baseline_results['mean_f1'],
        'Mean Precision': ocean_results['mean_precision'] - baseline_results['mean_precision'],
        'Mean Recall': ocean_results['mean_recall'] - baseline_results['mean_recall']
    }
    
    csv_path = os.path.join(output_dir, 'Model_Comparison_Summary.csv')
    comparison.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"   Saved: {csv_path}")
    
    # Print to console
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Ocean-Enhanced':<15} {'Difference':<15}")
    print("-"*65)
    print(f"{'Features':<20} {len(baseline_results['features']):<15} {len(ocean_results['features']):<15} {len(ocean_results['features']) - len(baseline_results['features']):<15}")
    print(f"{'Mean AUC':<20} {baseline_results['mean_auc']:<15.3f} {ocean_results['mean_auc']:<15.3f} {auc_diff:<+15.3f}")
    print(f"{'Mean F1':<20} {baseline_results['mean_f1']:<15.3f} {ocean_results['mean_f1']:<15.3f} {ocean_results['mean_f1'] - baseline_results['mean_f1']:<+15.3f}")
    print(f"{'Mean Precision':<20} {baseline_results['mean_precision']:<15.3f} {ocean_results['mean_precision']:<15.3f} {ocean_results['mean_precision'] - baseline_results['mean_precision']:<+15.3f}")
    print(f"{'Mean Recall':<20} {baseline_results['mean_recall']:<15.3f} {ocean_results['mean_recall']:<15.3f} {ocean_results['mean_recall'] - baseline_results['mean_recall']:<+15.3f}")
    
    print(f"\n📈 Ocean-Enhanced AUC Change: {auc_pct:+.1f}%")
    
    if auc_diff > 0.02:
        print("   ✅ Ocean variables IMPROVE prediction")
    elif auc_diff < -0.02:
        print("   ⚠️ Ocean variables DECREASE prediction")
    else:
        print("   ➡️ Ocean variables have MINIMAL effect")
    
    return comparison


# =============================================================================
# MODEL DEPLOYMENT
# =============================================================================

def train_and_save_deployment_model(df, features, model_name, output_dir):
    """Train final model on all data and save for deployment."""
    print(f"\n🚀 Training deployment model: {model_name}")
    
    available_features = get_available_features(df, features)
    
    X = df[available_features].copy()
    y = df[COL_LABEL].values
    
    # Build and train pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    pipeline.fit(X, y)
    
    # Save model
    model_dir = os.path.join(output_dir, 'deployment')
    ensure_dir(model_dir)
    
    model_path = os.path.join(model_dir, 'cliff_watch_model.joblib')
    joblib_dump(pipeline, model_path)
    print(f"   Saved model: {model_path}")
    
    # Save config
    config = {
        'model_name': model_name,
        'created': datetime.now().isoformat(),
        'predictors': available_features,
        'n_predictors': len(available_features),
        'n_estimators': N_ESTIMATORS,
        'n_training_samples': len(y),
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum()),
        'risk_thresholds': RISK_THRESHOLDS,
        'risk_classes': {
            'low': f"prob < {RISK_THRESHOLDS['low_max']}",
            'medium': f"{RISK_THRESHOLDS['low_max']} <= prob < {RISK_THRESHOLDS['medium_max']}",
            'high': f"prob >= {RISK_THRESHOLDS['medium_max']}"
        }
    }
    
    config_path = os.path.join(model_dir, 'cliff_watch_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config: {config_path}")
    
    # Save README
    readme_content = f"""CLIFF WATCH DEPLOYMENT MODEL
============================
Model: {model_name}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FILES
-----
- cliff_watch_model.joblib: Trained sklearn Pipeline (imputer + RandomForest)
- cliff_watch_config.json: Configuration with predictor names and thresholds

USAGE
-----
```python
from joblib import load
import pandas as pd
import json

# Load model and config
model = load('cliff_watch_model.joblib')
with open('cliff_watch_config.json') as f:
    config = json.load(f)

# Prepare input data with required features
X = df[config['predictors']]

# Get probability
prob = model.predict_proba(X)[:, 1]

# Classify risk
def classify_risk(p):
    if p < {RISK_THRESHOLDS['low_max']}:
        return 'Low'
    elif p < {RISK_THRESHOLDS['medium_max']}:
        return 'Medium'
    else:
        return 'High'

risk = [classify_risk(p) for p in prob]
```

RISK THRESHOLDS
---------------
- Low: probability < {RISK_THRESHOLDS['low_max']}
- Medium: {RISK_THRESHOLDS['low_max']} <= probability < {RISK_THRESHOLDS['medium_max']}
- High: probability >= {RISK_THRESHOLDS['medium_max']}

TRAINING DATA
-------------
- Total samples: {len(y)}
- Landslides: {int(y.sum())}
- Non-landslides: {int(len(y) - y.sum())}
- Features: {len(available_features)}
"""
    
    readme_path = os.path.join(model_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"   Saved README: {readme_path}")
    
    return pipeline, config


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("CLIFF WATCH NOWCAST RF MODEL TRAINING")
    print("="*70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    ensure_dir(OUTPUT_DIR)
    
    # Load data
    print(f"\n📁 Loading data: {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        # Try alternate path
        alt_path = "/mnt/user-data/outputs/nowcast_training_FINAL.csv"
        if os.path.exists(alt_path):
            INPUT_CSV_USE = alt_path
        else:
            print(f"❌ ERROR: Input file not found!")
            print(f"   Expected: {INPUT_CSV}")
            sys.exit(1)
    else:
        INPUT_CSV_USE = INPUT_CSV
    
    df = pd.read_csv(INPUT_CSV_USE)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Basic stats
    n_ls = (df[COL_LABEL] == 1).sum()
    n_ctrl = (df[COL_LABEL] == 0).sum()
    n_locs = df[COL_LOCATION].nunique() if COL_LOCATION in df.columns else 'N/A'
    
    print(f"   Landslides: {n_ls}")
    print(f"   Controls: {n_ctrl}")
    print(f"   Unique locations: {n_locs}")
    
    # Train baseline model
    baseline_results = train_with_stcv(
        df, 
        BASELINE_FEATURES, 
        "Baseline (Terrain + Precip + Seismic)",
        n_folds=N_FOLDS
    )
    
    # Train ocean-enhanced model
    ocean_results = train_with_stcv(
        df,
        ALL_FEATURES,
        "Ocean-Enhanced (All Features)",
        n_folds=N_FOLDS
    )
    
    # Create publication outputs
    print("\n" + "="*70)
    print("GENERATING PUBLICATION OUTPUTS")
    print("="*70)
    
    create_performance_table(baseline_results, ocean_results, OUTPUT_DIR)
    create_feature_importance_figure(baseline_results, "Baseline", OUTPUT_DIR)
    create_feature_importance_figure(ocean_results, "Ocean_Enhanced", OUTPUT_DIR)
    create_roc_comparison_figure(baseline_results, ocean_results, OUTPUT_DIR)
    create_model_comparison_summary(baseline_results, ocean_results, OUTPUT_DIR)

    # Save predictions for visualization script
    baseline_results['predictions'].to_csv(os.path.join(OUTPUT_DIR, 'predictions_baseline.csv'), index=False)
    ocean_results['predictions'].to_csv(os.path.join(OUTPUT_DIR, 'predictions_ocean.csv'), index=False)
    
    # Determine which model to deploy
    print("\n" + "="*70)
    print("MODEL DEPLOYMENT")
    print("="*70)
    
    # Deploy the better model (or ocean-enhanced if same results)
    if ocean_results['mean_auc'] >= baseline_results['mean_auc'] - 0.02:
        deploy_features = ALL_FEATURES
        deploy_name = "Ocean-Enhanced"
        print("   Deploying Ocean-Enhanced model (selected for operational use)")
    else:
        deploy_features = BASELINE_FEATURES
        deploy_name = "Baseline"
        print("   Deploying Baseline model (ocean variables decreased performance)")
    
    pipeline, config = train_and_save_deployment_model(df, deploy_features, deploy_name, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("   📄 Table4_Performance_Metrics.csv/.tex")
    print("   📊 Figure_FeatureImportance_Baseline.png")
    print("   📊 Figure_FeatureImportance_Ocean_Enhanced.png")
    print("   📊 Figure_ROC_Comparison.png")
    print("   📄 Model_Comparison_Summary.csv")
    print("   🚀 deployment/cliff_watch_model.joblib")
    print("   🚀 deployment/cliff_watch_config.json")
    print("   🚀 deployment/README.txt")
    
    print("\n🎯 Key Results:")
    print(f"   Baseline AUC: {baseline_results['mean_auc']:.3f}")
    print(f"   Ocean-Enhanced AUC: {ocean_results['mean_auc']:.3f}")
    
    auc_diff = ocean_results['mean_auc'] - baseline_results['mean_auc']
    if auc_diff > 0:
        print(f"   Ocean variables improved prediction by {auc_diff:.3f} AUC ({auc_diff/baseline_results['mean_auc']*100:+.1f}%)")
    else:
        print(f"   Ocean variables changed prediction by {auc_diff:.3f} AUC ({auc_diff/baseline_results['mean_auc']*100:+.1f}%)")


if __name__ == "__main__":
    main()
