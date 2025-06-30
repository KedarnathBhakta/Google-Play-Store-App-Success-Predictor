"""
Model Explainability Module for Google Play Store Apps Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import eli5
from sklearn.inspection import partial_dependence
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Comprehensive model explainability for Google Play Store app predictions"""
    
    def __init__(self, model, X_train, X_test, feature_names, model_name="Model"):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.model_name = model_name
        self.shap_explainer = None
        self.lime_explainer = None
        
    def create_shap_explainer(self):
        """Create SHAP explainer for the model"""
        print("Creating SHAP explainer...")
        
        if hasattr(self.model, 'feature_importances_'):
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
        
        return self.shap_explainer
    
    def explain_global_shap(self, sample_size=1000):
        """Generate global SHAP explanations"""
        print("Generating global SHAP explanations...")
        
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        sample_data = self.X_test.sample(min(sample_size, len(self.X_test)), random_state=42)
        shap_values = self.shap_explainer.shap_values(sample_data)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_data, show=False)
        plt.title(f'SHAP Summary Plot - {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'reports/figures/shap_summary_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
    
    def explain_local_shap(self, sample_idx=0):
        """Generate local SHAP explanations for a specific sample"""
        print(f"Generating local SHAP explanation for sample {sample_idx}...")
        
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        sample = self.X_test.iloc[sample_idx:sample_idx+1]
        shap_values = self.shap_explainer.shap_values(sample)
        
        # Force plot
        plt.figure(figsize=(12, 6))
        shap.force_plot(
            self.shap_explainer.expected_value, 
            shap_values, 
            sample, 
            show=False
        )
        plt.title(f'SHAP Force Plot - Sample {sample_idx} - {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'reports/figures/shap_force_sample_{sample_idx}_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
    
    def create_lime_explainer(self):
        """Create LIME explainer for the model"""
        print("Creating LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Low', 'High'] if hasattr(self.model, 'classes_') else None,
            mode='regression' if not hasattr(self.model, 'classes_') else 'classification'
        )
        
        return self.lime_explainer
    
    def explain_local_lime(self, sample_idx=0, num_features=10):
        """Generate local LIME explanations for a specific sample"""
        print(f"Generating local LIME explanation for sample {sample_idx}...")
        
        if self.lime_explainer is None:
            self.create_lime_explainer()
        
        sample = self.X_test.iloc[sample_idx]
        
        exp = self.lime_explainer.explain_instance(
            sample.values,
            self.model.predict,
            num_features=num_features
        )
        
        # Save explanation as HTML
        exp.save_to_file(f'reports/figures/lime_explanation_sample_{sample_idx}_{self.model_name}.html')
        
        return exp
    
    def explain_eli5(self):
        """Generate ELI5 explanations"""
        print("Generating ELI5 explanations...")
        
        if hasattr(self.model, 'feature_importances_'):
            eli5_explanation = eli5.explain_weights(self.model, feature_names=self.feature_names)
            
            with open(f'reports/figures/eli5_weights_{self.model_name}.html', 'w') as f:
                f.write(eli5.format_html(eli5_explanation))
        
        return eli5_explanation
    
    def generate_comprehensive_report(self, sample_indices=[0, 1, 2]):
        """Generate comprehensive explanation report"""
        print("Generating comprehensive explanation report...")
        
        report = {
            'model_name': self.model_name,
            'global_explanations': {},
            'local_explanations': {},
        }
        
        # Global explanations
        try:
            shap_values = self.explain_global_shap()
            report['global_explanations']['shap'] = 'Generated successfully'
        except Exception as e:
            report['global_explanations']['shap'] = f'Error: {str(e)}'
        
        # ELI5 explanations
        try:
            eli5_explanation = self.explain_eli5()
            report['global_explanations']['eli5'] = 'Generated successfully'
        except Exception as e:
            report['global_explanations']['eli5'] = f'Error: {str(e)}'
        
        # Local explanations
        for idx in sample_indices:
            try:
                local_shap = self.explain_local_shap(idx)
                local_lime = self.explain_local_lime(idx)
                report['local_explanations'][f'sample_{idx}'] = {
                    'shap': 'Generated successfully',
                    'lime': 'Generated successfully'
                }
            except Exception as e:
                report['local_explanations'][f'sample_{idx}'] = {
                    'error': str(e)
                }
        
        # Save report
        import json
        with open(f'results/explanation_report_{self.model_name}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Explanation report saved to results/explanation_report_{self.model_name}.json")
        return report


def main():
    """Example usage of the ModelExplainer"""
    print("Model Explainer Example")
    print("ModelExplainer class provides:")
    print("- SHAP explanations (global and local)")
    print("- LIME explanations")
    print("- ELI5 explanations")
    print("- Comprehensive explanation reports")


if __name__ == "__main__":
    main() 