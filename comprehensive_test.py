"""
Comprehensive Testing Suite for Data-Centric XAI Project
Tests all aspects to ensure complete documentation readiness
"""

import os
import sys
import pickle
import json
from datetime import datetime
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import DataProcessor
from model_manager import ModelManager
from model_trainer import ModelTrainer
from model_builder import ModelBuilder
from xai_analyzer import XAIAnalyzer
from phase3_orchestrator import run_phase3_augmentation
from phase4_orchestrator import run_phase4_training


class ComprehensiveTest:
    """Test all project components and log results"""
    
    def __init__(self):
        self.test_log = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_tested": [],
            "models_tested": [],
            "phase3_results": [],
            "phase4_results": [],
            "files_generated": {
                "models": [],
                "reports": [],
                "visualizations": [],
                "datasets": []
            },
            "errors": []
        }
        
    def print_header(self, text):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f" {text}")
        print("=" * 80)
        
    def print_status(self, status, message):
        """Print status message"""
        symbols = {"OK": "[OK]", "FAIL": "[FAIL]", "INFO": "[DATA]", "WARN": "[WARN]"}
        print(f"{symbols.get(status, '[?]')} {message}")
        
    def test_file_structure(self):
        """Test if all required directories and files exist"""
        self.print_header("TEST 1: FILE STRUCTURE VERIFICATION")
        
        required_dirs = [
            "saved_models",
            "training_history",
            "augmented_datasets",
            "results",
            "reports",
            "images"
        ]
        
        required_files = [
            "main.py",
            "data_processor.py",
            "model_builder.py",
            "model_trainer.py",
            "model_manager.py",
            "xai_analyzer.py",
            "error_extractor.py",
            "augmenter.py",
            "dataset_creator.py",
            "dataset_manager.py",
            "phase3_orchestrator.py",
            "phase4_orchestrator.py",
            "augmented_trainer.py",
            "results_comparator.py",
            "datacentric_report.py",
            "requirements.txt",
            "README.md",
            "DATA_CENTRIC_XAI.md"
        ]
        
        all_ok = True
        
        # Check directories
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                self.print_status("OK", f"Directory exists: {dir_name}/")
            else:
                self.print_status("FAIL", f"Directory missing: {dir_name}/")
                all_ok = False
                
        # Check files
        for file_name in required_files:
            if os.path.exists(file_name):
                self.print_status("OK", f"File exists: {file_name}")
            else:
                self.print_status("FAIL", f"File missing: {file_name}")
                all_ok = False
                
        return all_ok
        
    def test_dataset_loading(self):
        """Test all three datasets can be loaded"""
        self.print_header("TEST 2: DATASET LOADING")
        
        dp = DataProcessor()
        datasets = ["mnist", "fashion", "cifar10"]
        results = {}
        
        for dataset_name in datasets:
            try:
                (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
                info = dp.get_dataset_info(dataset_name)
                
                self.print_status("OK", f"{dataset_name.upper()}: {x_train.shape} train, {x_test.shape} test")
                self.print_status("INFO", f"  Classes: {info['classes']}, Shape: {info['shape']}")
                
                self.test_log["datasets_tested"].append(dataset_name)
                results[dataset_name] = {
                    "train_shape": str(x_train.shape),
                    "test_shape": str(x_test.shape),
                    "classes": info['classes']
                }
            except Exception as e:
                self.print_status("FAIL", f"{dataset_name}: {str(e)}")
                self.test_log["errors"].append(f"Dataset loading failed: {dataset_name} - {str(e)}")
                
        return results
        
    def test_model_architectures(self):
        """Test all three model architectures can be built"""
        self.print_header("TEST 3: MODEL ARCHITECTURE BUILDING")
        
        models = ["mobilenet_v2", "efficient_cnn", "resnet18"]
        dataset_configs = {
            "mnist": {"shape": (32, 32, 1), "classes": 10},
            "cifar10": {"shape": (32, 32, 3), "classes": 10}
        }
        
        results = {}
        
        for model_type in models:
            for dataset_name, config in dataset_configs.items():
                try:
                    builder = ModelBuilder(config["shape"], config["classes"])
                    model = builder.build_model(model_type)
                    
                    params = model.count_params()
                    self.print_status("OK", f"{model_type} for {dataset_name}: {params:,} parameters")
                    
                    if model_type not in results:
                        results[model_type] = {}
                    results[model_type][dataset_name] = params
                    
                    self.test_log["models_tested"].append(f"{model_type}_{dataset_name}")
                except Exception as e:
                    self.print_status("FAIL", f"{model_type} for {dataset_name}: {str(e)}")
                    self.test_log["errors"].append(f"Model build failed: {model_type}_{dataset_name} - {str(e)}")
                    
        return results
        
    def test_existing_models(self):
        """Check what models are already trained"""
        self.print_header("TEST 4: EXISTING TRAINED MODELS")
        
        mm = ModelManager()
        saved_models = mm.list_saved_models()
        
        if saved_models:
            self.print_status("INFO", f"Found {len(saved_models)} saved models:")
            for model_file in saved_models:
                model_path = os.path.join("saved_models", model_file)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                self.print_status("OK", f"  {model_file} ({size_mb:.2f} MB)")
                self.test_log["files_generated"]["models"].append(model_file)
        else:
            self.print_status("WARN", "No saved models found - will need to train")
            
        return saved_models
        
    def test_phase3_workflow(self, dataset="mnist", model_type="mobilenet_v2", quick=True):
        """Test Phase 3: Error-driven augmentation"""
        self.print_header(f"TEST 5: PHASE 3 WORKFLOW ({dataset} - {model_type})")
        
        try:
            dp = DataProcessor()
            mm = ModelManager()
            trainer = ModelTrainer(mm)
            
            # Load data
            self.print_status("INFO", "Loading dataset...")
            (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset)
            x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
            
            # Use subset for quick test
            if quick:
                x_train_norm = x_train_norm[:5000]
                y_train = y_train[:5000]
                x_test_norm = x_test_norm[:1000]
                y_test = y_test[:1000]
                self.print_status("INFO", "Using subset for quick test (5K train, 1K test)")
            
            # Load or train baseline
            self.print_status("INFO", "Loading baseline model...")
            baseline_model, _, loaded = trainer.train_or_load_model(
                dataset, model_type, force_retrain=False, epochs=3
            )
            
            if loaded:
                self.print_status("OK", "Baseline model loaded from disk")
            else:
                self.print_status("OK", "Baseline model trained successfully")
                
            # Evaluate baseline
            baseline_acc, _, _ = trainer.evaluate_model(baseline_model, x_test_norm, y_test)
            self.print_status("INFO", f"Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
            
            # Run Phase 3
            self.print_status("INFO", "Running Phase 3 augmentation...")
            x_aug, y_aug, num_errors, num_augmented = run_phase3_augmentation(
                baseline_model, x_train_norm, y_train, x_test_norm, y_test,
                dataset, model_type, multiplier=2, save=True
            )
            
            self.print_status("OK", f"Phase 3 complete: {num_errors} errors, {num_augmented} augmented samples")
            
            # Log results
            phase3_result = {
                "dataset": dataset,
                "model": model_type,
                "baseline_accuracy": float(baseline_acc),
                "errors_found": num_errors,
                "augmented_samples": num_augmented,
                "original_size": len(x_train_norm),
                "augmented_size": len(x_aug)
            }
            self.test_log["phase3_results"].append(phase3_result)
            
            # Check if dataset was saved
            aug_dataset_file = f"{model_type}_{dataset}_augmented.pkl"
            aug_dataset_path = os.path.join("augmented_datasets", aug_dataset_file)
            if os.path.exists(aug_dataset_path):
                size_mb = os.path.getsize(aug_dataset_path) / (1024 * 1024)
                self.print_status("OK", f"Augmented dataset saved: {aug_dataset_file} ({size_mb:.2f} MB)")
                self.test_log["files_generated"]["datasets"].append(aug_dataset_file)
            else:
                self.print_status("WARN", "Augmented dataset not found")
                
            return phase3_result, x_aug, y_aug
            
        except Exception as e:
            self.print_status("FAIL", f"Phase 3 failed: {str(e)}")
            self.test_log["errors"].append(f"Phase 3 workflow failed: {str(e)}")
            return None, None, None
            
    def test_phase4_workflow(self, x_aug, y_aug, dataset, model_type, baseline_acc, 
                            num_errors, num_augmented, original_size, quick=True):
        """Test Phase 4: Retrain and compare"""
        self.print_header(f"TEST 6: PHASE 4 WORKFLOW ({dataset} - {model_type})")
        
        try:
            # Run Phase 4
            self.print_status("INFO", "Running Phase 4 training...")
            epochs = 3 if quick else 10
            
            aug_model, aug_acc, improvement = run_phase4_training(
                x_aug, y_aug, dataset, model_type,
                baseline_acc, num_errors, num_augmented, original_size,
                epochs=epochs
            )
            
            self.print_status("OK", f"Phase 4 complete!")
            self.print_status("INFO", f"Baseline: {baseline_acc*100:.2f}%")
            self.print_status("INFO", f"Augmented: {aug_acc*100:.2f}%")
            self.print_status("INFO", f"Improvement: {improvement*100:+.2f}%")
            
            # Log results
            phase4_result = {
                "dataset": dataset,
                "model": model_type,
                "baseline_accuracy": float(baseline_acc),
                "augmented_accuracy": float(aug_acc),
                "improvement": float(improvement),
                "epochs_trained": epochs
            }
            self.test_log["phase4_results"].append(phase4_result)
            
            # Check generated files
            self.check_generated_files(dataset, model_type)
            
            return phase4_result
            
        except Exception as e:
            self.print_status("FAIL", f"Phase 4 failed: {str(e)}")
            self.test_log["errors"].append(f"Phase 4 workflow failed: {str(e)}")
            return None
            
    def check_generated_files(self, dataset, model_type):
        """Check what files were generated"""
        self.print_header("CHECKING GENERATED FILES")
        
        # Check augmented model
        aug_model_file = f"{model_type}_{dataset}_augmented.h5"
        aug_model_path = os.path.join("saved_models", aug_model_file)
        if os.path.exists(aug_model_path):
            size_mb = os.path.getsize(aug_model_path) / (1024 * 1024)
            self.print_status("OK", f"Augmented model: {aug_model_file} ({size_mb:.2f} MB)")
            self.test_log["files_generated"]["models"].append(aug_model_file)
            
        # Check training history
        history_file = f"{model_type}_{dataset}_augmented.pkl"
        history_path = os.path.join("training_history", history_file)
        if os.path.exists(history_path):
            self.print_status("OK", f"Training history: {history_file}")
            
        # Check comparison plot
        comparison_file = f"comparison_{model_type}_{dataset}.png"
        comparison_path = os.path.join("results", comparison_file)
        if os.path.exists(comparison_path):
            self.print_status("OK", f"Comparison plot: {comparison_file}")
            self.test_log["files_generated"]["visualizations"].append(comparison_file)
        else:
            self.print_status("WARN", f"Comparison plot not found: {comparison_file}")
            
        # Check reports
        report_files = [f for f in os.listdir("reports") if dataset in f and model_type in f]
        if report_files:
            latest_report = sorted(report_files)[-1]
            self.print_status("OK", f"Latest report: {latest_report}")
            self.test_log["files_generated"]["reports"].append(latest_report)
        else:
            self.print_status("WARN", "No reports found")
            
    def test_xai_features(self, dataset="mnist", model_type="mobilenet_v2"):
        """Test XAI visualization features"""
        self.print_header(f"TEST 7: XAI FEATURES ({dataset} - {model_type})")
        
        try:
            dp = DataProcessor()
            mm = ModelManager()
            
            # Load model and data
            model = mm.load_model(dataset, model_type)
            if model is None:
                self.print_status("WARN", "Model not found, skipping XAI test")
                return
                
            (_, _), (x_test, y_test) = dp.load_dataset(dataset)
            x_train_orig, x_test_norm = dp.apply_normalization(x_test, x_test)
            class_names = dp.get_class_names(dataset)
            
            # Test XAI analyzer
            xai = XAIAnalyzer(model, class_names)
            
            # Test evaluation
            self.print_status("INFO", "Testing comprehensive metrics...")
            accuracy, macro_f1 = xai.evaluate_full_metrics(x_test_norm[:100], y_test[:100])
            self.print_status("OK", f"Metrics: Accuracy={accuracy:.4f}, F1={macro_f1:.4f}")
            
            # Test Grad-CAM (just 1 sample)
            self.print_status("INFO", "Testing Grad-CAM visualization...")
            xai.grad_cam_analysis(x_test_norm[:1], y_test[:1], dataset, num_samples=1)
            self.print_status("OK", "Grad-CAM completed")
            
            # Check if XAI images were saved
            xai_images = [f for f in os.listdir("images") if dataset in f]
            if xai_images:
                self.print_status("OK", f"Found {len(xai_images)} XAI visualization images")
                self.test_log["files_generated"]["visualizations"].extend(xai_images[:5])
            
        except Exception as e:
            self.print_status("FAIL", f"XAI test failed: {str(e)}")
            self.test_log["errors"].append(f"XAI features test failed: {str(e)}")
            
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        self.print_header("TEST SUMMARY REPORT")
        
        print(f"\nTest Date: {self.test_log['test_date']}")
        print(f"\nDatasets Tested: {len(self.test_log['datasets_tested'])}")
        for ds in self.test_log['datasets_tested']:
            print(f"  - {ds}")
            
        print(f"\nModels Tested: {len(self.test_log['models_tested'])}")
        
        print(f"\nPhase 3 Tests Completed: {len(self.test_log['phase3_results'])}")
        for result in self.test_log['phase3_results']:
            print(f"  - {result['model']}_{result['dataset']}: {result['errors_found']} errors")
            
        print(f"\nPhase 4 Tests Completed: {len(self.test_log['phase4_results'])}")
        for result in self.test_log['phase4_results']:
            print(f"  - {result['model']}_{result['dataset']}: {result['improvement']*100:+.2f}% improvement")
            
        print(f"\nFiles Generated:")
        print(f"  Models: {len(self.test_log['files_generated']['models'])}")
        print(f"  Reports: {len(self.test_log['files_generated']['reports'])}")
        print(f"  Visualizations: {len(self.test_log['files_generated']['visualizations'])}")
        print(f"  Datasets: {len(self.test_log['files_generated']['datasets'])}")
        
        if self.test_log['errors']:
            print(f"\nErrors Encountered: {len(self.test_log['errors'])}")
            for error in self.test_log['errors']:
                print(f"  ! {error}")
        else:
            print("\n[OK] No errors encountered!")
            
        # Save log to file
        log_file = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.test_log, f, indent=2)
        print(f"\nDetailed test log saved to: {log_file}")
        
    def run_comprehensive_test(self, quick=True):
        """Run all tests"""
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE PROJECT TESTING SUITE")
        print(" For LaTeX Documentation Preparation")
        print("=" * 80)
        
        # Test 1: File structure
        self.test_file_structure()
        
        # Test 2: Dataset loading
        self.test_dataset_loading()
        
        # Test 3: Model architectures
        self.test_model_architectures()
        
        # Test 4: Existing models
        self.test_existing_models()
        
        # Test 5 & 6: Phase 3 & 4 workflow (MNIST only for quick test)
        phase3_result, x_aug, y_aug = self.test_phase3_workflow(
            dataset="mnist", 
            model_type="mobilenet_v2",
            quick=quick
        )
        
        if phase3_result and x_aug is not None:
            self.test_phase4_workflow(
                x_aug, y_aug, 
                "mnist", "mobilenet_v2",
                phase3_result["baseline_accuracy"],
                phase3_result["errors_found"],
                phase3_result["augmented_samples"],
                phase3_result["original_size"],
                quick=quick
            )
            
        # Test 7: XAI features
        self.test_xai_features("mnist", "mobilenet_v2")
        
        # Final summary
        self.generate_test_summary()


if __name__ == "__main__":
    tester = ComprehensiveTest()
    
    # Quick test (fast, uses subsets)
    print("\nRunning QUICK test mode (uses data subsets)...")
    print("For full test, edit script and set quick=False\n")
    
    tester.run_comprehensive_test(quick=True)
    
    print("\n" + "=" * 80)
    print(" TESTING COMPLETE!")
    print(" Review the output above and check generated files")
    print("=" * 80)
