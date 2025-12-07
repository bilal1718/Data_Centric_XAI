import os
import pandas as pd
from datetime import datetime


class DataCentricReport:
    """Generate comprehensive report of the data-centric workflow"""
    
    def generate_report(self, dataset_name, model_type, 
                       baseline_acc, augmented_acc,
                       num_errors, num_augmented, 
                       original_train_size, final_train_size):
        """
        Create detailed report of the complete workflow
        
        Args:
            dataset_name: Dataset used
            model_type: Model architecture
            baseline_acc: Original accuracy
            augmented_acc: Accuracy after augmentation
            num_errors: Number of error samples found
            num_augmented: Total augmented samples created
            original_train_size: Original training set size
            final_train_size: Final augmented training set size
        
        Returns:
            report: String containing full report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        improvement = augmented_acc - baseline_acc
        baseline_error = 1 - baseline_acc
        augmented_error = 1 - augmented_acc
        error_reduction = baseline_error - augmented_error
        
        if baseline_error > 0:
            relative_error_reduction = (error_reduction / baseline_error) * 100
        else:
            relative_error_reduction = 0
        
        # Determine success status
        if improvement > 0.001:
            status = "SUCCESS"
            conclusion = "Data-centric approach successfully improved model accuracy"
        elif improvement >= 0:
            status = "STABLE"
            conclusion = "Accuracy maintained (dataset may already be optimal)"
        else:
            status = "NEEDS REVIEW"
            conclusion = "Accuracy decreased - review augmentation strategy"
        
        report = f"""
{'=' * 70}
DATA-CENTRIC XAI WORKFLOW - COMPREHENSIVE REPORT
{'=' * 70}

PROJECT:    Error-Driven Dataset Augmentation
DATASET:    {dataset_name.upper()}
MODEL:      {model_type}
STATUS:     {status}
DATE:       {timestamp}

{'=' * 70}
PHASE 1: BASELINE TRAINING
{'=' * 70}
Training Set Size:          {original_train_size:,}
Baseline Test Accuracy:     {baseline_acc:.4f} ({baseline_acc*100:.2f}%)
Baseline Error Rate:        {baseline_error:.4f} ({baseline_error*100:.2f}%)

{'=' * 70}
PHASE 2: XAI ERROR ANALYSIS
{'=' * 70}
Misclassified Samples:      {num_errors}
Test Set Size:              10,000 (typical)
Error Rate on Test:         {num_errors / 10000 * 100:.2f}%
Samples for Augmentation:   {num_errors}

{'=' * 70}
PHASE 3: ERROR-DRIVEN AUGMENTATION
{'=' * 70}
Original Error Samples:     {num_errors}
Augmented Samples Created:  {num_augmented}
Augmentation Multiplier:    {num_augmented // num_errors if num_errors > 0 else 0}x
Dataset Size Increase:      +{num_augmented} (+{num_augmented/original_train_size*100:.2f}%)
Final Training Set Size:    {final_train_size:,}

Strategy Applied:
  - Rotation: ±15 degrees
  - Shifts: ±10-15%
  - Zoom: ±10%
  - Shear: ±10% (grayscale only)
  - Horizontal Flip: Yes (CIFAR-10 only)

{'=' * 70}
PHASE 4: RETRAIN ON AUGMENTED DATA
{'=' * 70}
Augmented Test Accuracy:    {augmented_acc:.4f} ({augmented_acc*100:.2f}%)
Augmented Error Rate:       {augmented_error:.4f} ({augmented_error*100:.2f}%)

{'=' * 70}
RESULTS: DATA-CENTRIC IMPROVEMENT ANALYSIS
{'=' * 70}

Accuracy Metrics:
  Baseline Accuracy:        {baseline_acc:.4f} ({baseline_acc*100:.2f}%)
  Augmented Accuracy:       {augmented_acc:.4f} ({augmented_acc*100:.2f}%)
  Absolute Improvement:     {improvement:+.4f} ({improvement*100:+.2f}%)
  Relative Improvement:     {improvement/baseline_acc*100:+.2f}%

Error Metrics:
  Baseline Error:           {baseline_error:.4f} ({baseline_error*100:.2f}%)
  Augmented Error:          {augmented_error:.4f} ({augmented_error*100:.2f}%)
  Error Reduction:          {error_reduction:.4f} ({error_reduction*100:.2f}%)
  Relative Error Reduction: {relative_error_reduction:+.1f}%

Data Quality Impact:
  Original Training Data:   {original_train_size:,} samples
  Targeted Augmentation:    {num_augmented:,} samples (+{num_augmented/original_train_size*100:.1f}%)
  Samples per Error:        {num_augmented // num_errors if num_errors > 0 else 0}
  Data Efficiency Gain:     {improvement/(num_augmented/original_train_size)*100:.1f}% per 1% data increase

{'=' * 70}
CONCLUSION
{'=' * 70}

{conclusion}

Key Findings:
  1. XAI successfully identified {num_errors} error-prone samples
  2. Targeted augmentation created {num_augmented:,} enhanced samples
  3. Model accuracy changed by {improvement*100:+.2f}%
  4. Error rate changed by {-error_reduction*100:+.2f}%

Recommendations:
"""
        
        # Add recommendations based on results
        if improvement > 0.005:
            report += f"""  + Excellent results - data-centric approach validated
  + Consider applying to other models/datasets
  + Document this methodology for future projects
"""
        elif improvement > 0:
            report += f"""  + Positive improvement demonstrated
  + Consider increasing augmentation multiplier
  + Try additional augmentation techniques
"""
        elif improvement >= -0.001:
            report += f"""  • Dataset may already be near-optimal
  • Consider more aggressive augmentation
  • Validate with different random seeds
"""
        else:
            report += f"""  ! Review augmentation parameters
  ! Increase training epochs
  ! Check for data quality issues
  ! Validate preprocessing consistency
"""
        
        report += f"\n{'=' * 70}\n"
        
        print(report)
        
        # Save report to file
        os.makedirs("reports", exist_ok=True)
        report_file = os.path.join("reports", 
                                   f"datacentric_report_{model_type}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved: {report_file}")
        
        # Also save as CSV for easy analysis
        csv_data = {
            'Timestamp': [timestamp],
            'Dataset': [dataset_name],
            'Model': [model_type],
            'Baseline_Accuracy': [baseline_acc],
            'Augmented_Accuracy': [augmented_acc],
            'Improvement': [improvement],
            'Improvement_Percent': [improvement * 100],
            'Baseline_Error': [baseline_error],
            'Augmented_Error': [augmented_error],
            'Error_Reduction': [error_reduction],
            'Relative_Error_Reduction': [relative_error_reduction],
            'Num_Errors': [num_errors],
            'Num_Augmented': [num_augmented],
            'Original_Train_Size': [original_train_size],
            'Final_Train_Size': [final_train_size],
            'Status': [status]
        }
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join("reports", "datacentric_results.csv")
        
        # Append to existing CSV or create new
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        
        print(f"Results appended to: {csv_file}")
        
        return report
