"""
Complete Training Pipeline untuk Receipt Parser
Menggunakan Random Forest Classifier

Pipeline:
1. Load annotated data
2. Extract features
3. Train model dengan cross-validation
4. Evaluate performance
5. Save model
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder

# Import dari ml_receipt_parser
from ml_receipt_parser import MLReceiptParser, PureFeatureExtractor # RF Only
# from ml_receipt_parser import MLReceiptParser, FeatureExtractor     # RF + Rule-Based


class TrainingPipeline:
    """Complete training pipeline untuk Receipt Parser"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize training pipeline
        
        Args:
            config: Dictionary konfigurasi training
        """
        self.config = config or {}
        self.parser = MLReceiptParser()
        self.results = {}
    
    def load_data(self, annotation_dir: str) -> List[Dict]:
        """
        Load semua annotated data dari directory
        
        Args:
            annotation_dir: directory berisi annotated JSON files
            
        Returns:
            List of annotated data
        """
        print(f"\n{'='*60}")
        print("LOADING ANNOTATED DATA")
        print('='*60)
        
        data = []
        json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(annotation_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    annotated_data = json.load(f)
                
                # Validate structure
                if 'ocr_result' in annotated_data and 'annotations' in annotated_data:
                    data.append(annotated_data)
                    print(f"✓ Loaded: {filename}")
                else:
                    print(f"✗ Invalid format: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        
        print(f"\nTotal files loaded: {len(data)}")
        
        # Statistics
        if data:
            total_annotations = sum(len(d['annotations']) for d in data)
            print(f"Total annotations: {total_annotations}")
            
            # Count labels
            label_counts = {}
            for d in data:
                for ann in d['annotations']:
                    label = ann['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            print("\nLabel distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label:12s}: {count:4d} ({count/total_annotations*100:.1f}%)")
        
        return data
    
    def prepare_features(self, annotated_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare feature matrix dan labels
        
        Args:
            annotated_data: list of annotated data
            
        Returns:
            X (DataFrame): features
            y (array): labels
        """
        print(f"\n{'='*60}")
        print("EXTRACTING FEATURES")
        print('='*60)
        
        X, y = self.parser.prepare_training_data(annotated_data)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {len(y)}")
        
        return X, y
    
    def train_baseline(self, X: pd.DataFrame, y: np.ndarray, 
                       test_size: float = 0.2, random_state: int = 42):
        """
        Train baseline Random Forest model
        
        Args:
            X: feature matrix
            y: labels
            test_size: test split ratio
            random_state: random seed
        """
        print(f"\n{'='*60}")
        print("TRAINING BASELINE MODEL")
        print('='*60)
        
        # Encode labels
        y_encoded = self.parser.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train model
        self.parser.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("\nTraining...")
        self.parser.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.parser.model.score(X_train, y_train)
        test_score = self.parser.model.score(X_test, y_test)
        
        print(f"\nTrain Accuracy: {train_score:.4f}")
        print(f"Test Accuracy:  {test_score:.4f}")
        
        # Predictions
        y_pred = self.parser.model.predict(X_test)
        
        # Store results
        self.results['baseline'] = {
            'train_score': train_score,
            'test_score': test_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train
        }
        
        return train_score, test_score
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: np.ndarray,
                             cv: int = 5):
        """
        Hyperparameter tuning menggunakan GridSearchCV
        
        Args:
            X_train: training features
            y_train: training labels
            cv: number of cross-validation folds
        """
        print(f"\n{'='*60}")
        print("HYPERPARAMETER TUNING")
        print('='*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nRunning GridSearchCV with {cv}-fold cross-validation...")
        print("This may take a while...\n")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n{'='*60}")
        print("BEST PARAMETERS")
        print('='*60)
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest CV Score: {grid_search.best_score_:.4f}")
        
        # Update model dengan best params
        self.parser.model = grid_search.best_estimator_
        
        self.results['tuning'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_params_
    
    def cross_validation(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5):
        """
        Perform cross-validation
        
        Args:
            X: feature matrix
            y: labels (encoded)
            cv: number of folds
        """
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({cv}-fold)")
        print('='*60)
        
        scores = cross_val_score(
            self.parser.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        print(f"\nCross-validation scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        self.results['cv'] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        return scores
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print('='*60)
        
        y_test = self.results['baseline']['y_test']
        y_pred = self.results['baseline']['y_pred']
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.parser.label_encoder.classes_,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        print("\nPer-class Metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        for i, label in enumerate(self.parser.label_encoder.classes_):
            print(f"{label:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        
        # Store evaluation results
        self.results['evaluation'] = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    def feature_importance_analysis(self):
        """
        Analisis feature importance
        """
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print('='*60)
        
        feature_importance = pd.DataFrame({
            'feature': self.parser.feature_names,
            'importance': self.parser.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))
        
        self.results['feature_importance'] = feature_importance
        
        return feature_importance
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: path untuk menyimpan model
        """
        print(f"\n{'='*60}")
        print("SAVING MODEL")
        print('='*60)
        
        self.parser.save_model(model_path)
        
        # Save training results
        results_path = model_path.replace('.pkl', '_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if key == 'baseline':
                results_json[key] = {
                    'train_score': float(value['train_score']),
                    'test_score': float(value['test_score'])
                }
            elif key == 'cv':
                results_json[key] = {
                    'mean': float(value['mean']),
                    'std': float(value['std']),
                    'scores': value['scores'].tolist()
                }
            elif key == 'tuning':
                results_json[key] = {
                    'best_params': value['best_params'],
                    'best_score': float(value['best_score'])
                }
            elif key == 'evaluation':
                results_json[key] = {
                    'confusion_matrix': value['confusion_matrix'].tolist(),
                    'precision': value['precision'].tolist(),
                    'recall': value['recall'].tolist(),
                    'f1': value['f1'].tolist(),
                    'support': value['support'].tolist()
                }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Results saved to: {results_path}")
    
    def plot_results(self, output_dir: str):
        """
        Generate visualization plots
        
        Args:
            output_dir: directory untuk menyimpan plots
        """
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print('='*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = self.results['evaluation']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.parser.label_encoder.classes_,
                   yticklabels=self.parser.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to: {cm_path}")
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        top_features = self.results['feature_importance'].head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        fi_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved to: {fi_path}")
        
        # 3. Per-class Performance
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame({
            'Precision': self.results['evaluation']['precision'],
            'Recall': self.results['evaluation']['recall'],
            'F1-Score': self.results['evaluation']['f1']
        }, index=self.parser.label_encoder.classes_)
        
        metrics_df.plot(kind='bar', rot=0)
        plt.title('Per-class Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        plt.tight_layout()
        perf_path = os.path.join(output_dir, 'per_class_performance.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Per-class performance plot saved to: {perf_path}")


def run_complete_pipeline(annotation_dir: str, model_output_path: str,
                          plot_output_dir: str, tune_hyperparams: bool = False):
    """
    Run complete training pipeline
    
    Args:
        annotation_dir: directory berisi annotated data
        model_output_path: path untuk save model
        plot_output_dir: directory untuk save plots
        tune_hyperparams: whether to perform hyperparameter tuning
    """
    pipeline = TrainingPipeline()
    
    # Step 1: Load data
    annotated_data = pipeline.load_data(annotation_dir)
    
    if not annotated_data:
        print("\nError: No annotated data found!")
        return
    
    # Step 2: Extract features
    X, y = pipeline.prepare_features(annotated_data)
    
    # Step 3: Train baseline model
    pipeline.train_baseline(X, y)
    
    # Step 4: Hyperparameter tuning (optional)
    if tune_hyperparams:
        X_train = pipeline.results['baseline']['X_train']
        y_train = pipeline.results['baseline']['y_train']
        pipeline.hyperparameter_tuning(X_train, y_train, cv=5)
        
        # Re-evaluate dengan tuned model
        X_test = pipeline.results['baseline']['X_test']
        y_test = pipeline.results['baseline']['y_test']
        y_pred = pipeline.parser.model.predict(X_test)
        pipeline.results['baseline']['y_pred'] = y_pred
    
    # Step 5: Cross-validation
    y_encoded = pipeline.parser.label_encoder.transform(y)
    pipeline.cross_validation(X, y_encoded, cv=5)
    
    # Step 6: Evaluate model
    pipeline.evaluate_model()
    
    # Step 7: Feature importance analysis
    pipeline.feature_importance_analysis()
    
    # Step 8: Save model
    pipeline.save_model(model_output_path)
    
    # Step 9: Generate plots
    pipeline.plot_results(plot_output_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE COMPLETED ✅")
    print('='*60)
    print(f"Model saved to: {model_output_path}")
    print(f"Plots saved to: {plot_output_dir}")
    print(f"\nYou can now use the trained model for inference:")
    print(f"  parser = MLReceiptParser(model_path='{model_output_path}')")
    print(f"  result = parser.parse(ocr_result)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Default paths
    ANNOTATION_DIR = "annotations"
    MODEL_PATH = "models/receipt_parser_rf.pkl"
    PLOT_DIR = "plots"
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       RECEIPT PARSER - RANDOM FOREST TRAINING PIPELINE        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if annotation directory exists
    if not os.path.exists(ANNOTATION_DIR):
        print(f"Error: Annotation directory '{ANNOTATION_DIR}' not found!")
        print("\nPlease create annotated data first using annotation_tool.py:")
        print(f"  python annotation_tool.py batch output/ {ANNOTATION_DIR}/")
        sys.exit(1)
    
    # Ask user for hyperparameter tuning
    tune = input("\nPerform hyperparameter tuning? (y/n, default=n): ").strip().lower()
    tune_hyperparams = tune == 'y'
    
    if tune_hyperparams:
        print("\n⚠️  Hyperparameter tuning will take significantly longer!")
    
    # Run pipeline
    try:
        run_complete_pipeline(
            annotation_dir=ANNOTATION_DIR,
            model_output_path=MODEL_PATH,
            plot_output_dir=PLOT_DIR,
            tune_hyperparams=tune_hyperparams
        )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
