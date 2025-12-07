"""
Automated Batch Training Script for LaTeX Documentation
Trains all model/dataset combinations systematically
"""

import os
import sys
import json
import csv
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(__file__))

from data_processor import DataProcessor
from model_manager import ModelManager
from model_trainer import ModelTrainer
from phase3_orchestrator import run_phase3_augmentation
from phase4_orchestrator import run_phase4_training


class BatchTrainer:
    """Automated training for all combinations"""
    
    def __init__(self):
        self.results = []
        self.datasets = ["mnist", "fashion", "cifar10"]
        self.models = ["mobilenet_v2", "efficient_cnn", "resnet18"]
        
    def train_baseline_models(self, epochs=10):
        """Train all 9 baseline models"""
        print("\n" + "=" * 80)
        print(" PHASE 1: TRAINING ALL BASELINE MODELS")
        print("=" * 80)
        print(f"\nTotal combinations: {len(self.datasets)} × {len(self.models)} = 9 models")
        print(f"Epochs per model: {epochs}")
        print(f"Estimated time: ~2-3 hours\n")
        
        input("Press Enter to start training...")
        
        dp = DataProcessor()
        mm = ModelManager()
        trainer = ModelTrainer(mm)
        
        for i, dataset in enumerate(self.datasets, 1):
            for j, model_type in enumerate(self.models, 1):
                combo_num = (i-1) * len(self.models) + j
                
                print(f"\n{'='*80}")
                print(f" [{combo_num}/9] Training: {model_type} on {dataset}")
                print(f"{'='*80}")
                
                start_time = time.time()
                
                try:
                    # Load data
                    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset)
                    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
                    
                    # Train model
                    model, history, loaded = trainer.train_or_load_model(
                        dataset, model_type, force_retrain=False, epochs=epochs
                    )
                    
                    # Evaluate
                    acc, loss, _ = trainer.evaluate_model(model, x_test_norm, y_test)
                    
                    training_time = (time.time() - start_time) / 60  # minutes
                    
                    result = {
                        "dataset": dataset,
                        "model": model_type,
                        "type": "baseline",
                        "accuracy": float(acc),
                        "loss": float(loss),
                        "parameters": model.count_params(),
                        "training_time_min": round(training_time, 2),
                        "epochs": epochs,
                        "status": "loaded" if loaded else "trained"
                    }
                    
                    self.results.append(result)
                    
                    print(f"\nSuccess!")
                    print(f"   Accuracy: {acc*100:.2f}%")
                    print(f"   Loss: {loss:.4f}")
                    print(f"   Time: {training_time:.1f} min")
                    print(f"   Status: {result['status']}")
                    
                except Exception as e:
                    print(f"\nFailed: {str(e)}")
                    self.results.append({
                        "dataset": dataset,
                        "model": model_type,
                        "type": "baseline",
                        "status": "failed",
                        "error": str(e)
                    })
                    
        self._save_results("baseline_training_results")
        
    def train_augmented_models(self, epochs=10, multiplier=2):
        """Run complete Phase 3-4 workflow for all combinations"""
        print("\n" + "=" * 80)
        print(" PHASE 2: DATA-CENTRIC AUGMENTATION FOR ALL MODELS")
        print("=" * 80)
        print(f"\nTotal combinations: 9 workflows")
        print(f"Augmentation multiplier: {multiplier}x")
        print(f"Retraining epochs: {epochs}")
        print(f"Estimated time: ~3-4 hours\n")
        
        input("Press Enter to start workflows...")
        
        dp = DataProcessor()
        mm = ModelManager()
        trainer = ModelTrainer(mm)
        
        for i, dataset in enumerate(self.datasets, 1):
            for j, model_type in enumerate(self.models, 1):
                combo_num = (i-1) * len(self.models) + j
                
                print(f"\n{'='*80}")
                print(f" [{combo_num}/9] Data-Centric Workflow: {model_type} on {dataset}")
                print(f"{'='*80}")
                
                start_time = time.time()
                
                try:
                    # Load data
                    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset)
                    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
                    
                    # Get baseline model
                    baseline_model, _, _ = trainer.train_or_load_model(
                        dataset, model_type, force_retrain=False
                    )
                    
                    # Evaluate baseline
                    baseline_acc, _, _ = trainer.evaluate_model(
                        baseline_model, x_test_norm, y_test
                    )
                
                    print(f"\nBaseline: {baseline_acc*100:.2f}%")
                    
                    # Phase 3: Augmentation
                    print(f"\nRunning Phase 3: Error-driven augmentation...")
                    x_aug, y_aug, num_errors, num_augmented = run_phase3_augmentation(
                        baseline_model, x_train_norm, y_train, 
                        x_test_norm, y_test, dataset, model_type,
                        multiplier=multiplier, save=True
                    )
                    
                    # Phase 4: Retrain & Compare
                    print(f"\nRunning Phase 4: Retraining on augmented data...")
                    aug_model, aug_acc, improvement = run_phase4_training(
                        x_aug, y_aug, dataset, model_type,
                        baseline_acc, num_errors, num_augmented,
                        len(x_train), epochs=epochs
                    )
                    
                    workflow_time = (time.time() - start_time) / 60
                    
                    result = {
                        "dataset": dataset,
                        "model": model_type,
                        "type": "augmented",
                        "baseline_accuracy": float(baseline_acc),
                        "augmented_accuracy": float(aug_acc),
                        "improvement": float(improvement),
                        "errors_found": num_errors,
                        "augmented_samples": num_augmented,
                        "original_size": len(x_train),
                        "augmented_size": len(x_aug),
                        "growth_percent": round((num_augmented / len(x_train)) * 100, 2),
                        "workflow_time_min": round(workflow_time, 2),
                        "epochs": epochs,
                        "multiplier": multiplier,
                        "status": "success"
                    }
                    
                    self.results.append(result)
                    
                    print(f"\nSuccess!")
                    print(f"   Baseline: {baseline_acc*100:.2f}%")
                    print(f"   Augmented: {aug_acc*100:.2f}%")
                    print(f"   Improvement: {improvement*100:+.2f}%")
                    print(f"   Time: {workflow_time:.1f} min")
                    
                except Exception as e:
                    print(f"\nFailed: {str(e)}")
                    self.results.append({
                        "dataset": dataset,
                        "model": model_type,
                        "type": "augmented",
                        "status": "failed",
                        "error": str(e)
                    })
                    
        self._save_results("augmented_training_results")
        
    def _save_results(self, prefix):
        """Save results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = f"{prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {json_file}")
        
        # Save CSV for LaTeX
        csv_file = f"{prefix}_{timestamp}.csv"
        if self.results:
            keys = self.results[0].keys()
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"CSV saved to: {csv_file}")
            
    def generate_latex_tables(self):
        """Generate ready-to-use LaTeX table code"""
        print("\n" + "=" * 80)
        print(" GENERATING LATEX TABLES")
        print("=" * 80)
        
        # Separate baseline and augmented results
        baseline_results = [r for r in self.results if r.get("type") == "baseline" and r.get("status") != "failed"]
        augmented_results = [r for r in self.results if r.get("type") == "augmented" and r.get("status") != "failed"]
        
        latex_file = f"latex_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        
        with open(latex_file, 'w') as f:
            # Table 1: Baseline Performance
            f.write("% Table 1: Baseline Model Performance\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Baseline Model Performance Across Datasets}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write("Dataset & Model & Accuracy & Loss & Parameters & Training Time \\\\\n")
            f.write("\\midrule\n")
            
            for dataset in self.datasets:
                f.write(f"\\multirow{{3}}{{*}}{{{dataset.upper()}}}\n")
                for model in self.models:
                    result = next((r for r in baseline_results 
                                 if r["dataset"] == dataset and r["model"] == model), None)
                    if result:
                        f.write(f" & {model.replace('_', ' ').title()} & "
                               f"{result['accuracy']*100:.2f}\\% & "
                               f"{result['loss']:.4f} & "
                               f"{result['parameters']:,} & "
                               f"{result['training_time_min']:.0f} min \\\\\n")
                    else:
                        f.write(f" & {model.replace('_', ' ').title()} & -- & -- & -- & -- \\\\\n")
                if dataset != self.datasets[-1]:
                    f.write("\\midrule\n")
                    
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Table 2: Data-Centric Improvement
            f.write("% Table 2: Data-Centric Improvement Analysis\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Data-Centric Approach: Accuracy Improvement}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write("Dataset & Model & Baseline & Augmented & Improvement & Error Reduction \\\\\n")
            f.write("\\midrule\n")
            
            for dataset in self.datasets:
                f.write(f"\\multirow{{3}}{{*}}{{{dataset.upper()}}}\n")
                for model in self.models:
                    result = next((r for r in augmented_results 
                                 if r["dataset"] == dataset and r["model"] == model), None)
                    if result:
                        error_reduction = (result['baseline_accuracy'] - result['augmented_accuracy']) * 100
                        f.write(f" & {model.replace('_', ' ').title()} & "
                               f"{result['baseline_accuracy']*100:.2f}\\% & "
                               f"{result['augmented_accuracy']*100:.2f}\\% & "
                               f"{result['improvement']*100:+.2f}\\% & "
                               f"{error_reduction:+.2f}\\% \\\\\n")
                    else:
                        f.write(f" & {model.replace('_', ' ').title()} & -- & -- & -- & -- \\\\\n")
                if dataset != self.datasets[-1]:
                    f.write("\\midrule\n")
                    
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
        print(f"LaTeX tables generated: {latex_file}")
        print(f"   Copy and paste into your .tex document")
        
    def run_complete_batch(self, baseline_epochs=10, augmented_epochs=10, multiplier=2):
        """Run complete training pipeline"""
        print("\n" + "=" * 80)
        print(" COMPLETE BATCH TRAINING FOR LATEX DOCUMENTATION")
        print("=" * 80)
        print(f"\nThis will train:")
        print(f"  - 9 baseline models ({baseline_epochs} epochs each)")
        print(f"  - 9 augmented models ({augmented_epochs} epochs each)")
        print(f"  - Total: 18 models + 9 augmented datasets")
        print(f"\nEstimated total time: 5-7 hours")
        print(f"\nMake sure you have sufficient disk space (~2-3 GB)")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Batch training cancelled.")
            return
            
        overall_start = time.time()
        
        # Phase 1: Baseline models
        self.train_baseline_models(epochs=baseline_epochs)
        
        # Phase 2: Augmented models
        self.train_augmented_models(epochs=augmented_epochs, multiplier=multiplier)
        
        # Generate LaTeX tables
        self.generate_latex_tables()
        
        total_time = (time.time() - overall_start) / 3600  # hours
        
        print("\n" + "=" * 80)
        print(" BATCH TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nTotal time: {total_time:.2f} hours")
        print(f"Total models trained: {len([r for r in self.results if r.get('status') != 'failed'])}")
        print(f"Failed: {len([r for r in self.results if r.get('status') == 'failed'])}")
        print(f"\nAll results saved in current directory.")
        print(f"Check JSON/CSV files and LaTeX tables.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch training for LaTeX documentation")
    parser.add_argument("--mode", choices=["baseline", "augmented", "full"], default="full",
                       help="Training mode: baseline only, augmented only, or full pipeline")
    parser.add_argument("--baseline-epochs", type=int, default=10,
                       help="Epochs for baseline training (default: 10)")
    parser.add_argument("--augmented-epochs", type=int, default=10,
                       help="Epochs for augmented training (default: 10)")
    parser.add_argument("--multiplier", type=int, default=2,
                       help="Augmentation multiplier (default: 2)")
    
    args = parser.parse_args()
    
    trainer = BatchTrainer()
    
    if args.mode == "baseline":
        trainer.train_baseline_models(epochs=args.baseline_epochs)
        trainer.generate_latex_tables()
    elif args.mode == "augmented":
        trainer.train_augmented_models(epochs=args.augmented_epochs, multiplier=args.multiplier)
        trainer.generate_latex_tables()
    else:  # full
        trainer.run_complete_batch(
            baseline_epochs=args.baseline_epochs,
            augmented_epochs=args.augmented_epochs,
            multiplier=args.multiplier
        )
