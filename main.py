

import os
import sys
import logging
from model_manager import ModelManager
from data_processor import DataProcessor
from model_builder import ModelBuilder
from model_trainer import ModelTrainer
from xai_analyzer import XAIAnalyzer
from phase3_orchestrator import run_phase3_augmentation
from phase4_orchestrator import run_phase4_training

# Configure logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def print_banner():
   
    print("\n" + "=" * 80)
    print(" Model WITH EXPLAINABLE AI (XAI)")
    print("=" * 80)
    print("A Comprehensive Framework for CNN Architecture Evaluation")
    print("with Integrated Model Interpretability")
    print("=" * 80)

def print_menu():
   
    print("\n MAIN MENU:")
    print("1.  Train/Load Single Model with XAI Analysis")
    print("2.  Compare All Models (Consolidated Analysis) [Not Available]")
    print("3.  Run XAI Analysis on Pre-trained Model")
    print("4.  Generate Research Summary Report [Not Available]")
    print("5.  List Available Pre-trained Models")
    print("6.  Quick Start (Complete Pipeline)")
    print("7.  Data-Centric Workflow (Error-Driven Augmentation) �")
    print("8.  Exit")

def get_user_choice(prompt, valid_options, default=None):
    """Get validated user input"""
    while True:
        try:
            choice = input(prompt).strip()
            if not choice and default is not None:
                return default
            if choice in valid_options:
                return choice
            print(f" Invalid choice. Please choose from {valid_options}")
        except KeyboardInterrupt:
            print("\n Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f" Error: {e}")

def single_model_analysis():
    
    print("\n" + "=" * 60)
    print(" SINGLE MODEL ANALYSIS WITH XAI")
    print("=" * 60)
    
    # Dataset selection
    print("\n Available Datasets:")
    print("1. MNIST (Handwritten digits)")
    print("2. Fashion-MNIST (Fashion items)")
    print("3. CIFAR-10 (Color objects)")
    
    ds_choice = get_user_choice("Select dataset (1-3) [default 3]: ", ["1", "2", "3"], "3")
    ds_map = {"1": "mnist", "2": "fashion", "3": "cifar10"}
    dataset_name = ds_map[ds_choice]
    
    # Model selection
    print("\n Available Models:")
    print("1. MobileNetV2 (Lightweight)")
    print("2. Efficient CNN (Multi-scale)")
    print("3. ResNet18 (Residual learning)")
    
    m_choice = get_user_choice("Select model (1-3) [default 1]: ", ["1", "2", "3"], "1")
    m_map = {"1": "mobilenet_v2", "2": "efficient_cnn", "3": "resnet18"}
    model_type = m_map[m_choice]
    
    # Training options
    print("\n Training Options:")
    print("1. Use pre-trained model if available")
    print("2. Force retrain (even if pre-trained exists)")
    
    t_choice = get_user_choice("Select option (1-2) [default 1]: ", ["1", "2"], "1")
    force_retrain = (t_choice == "2")
    
    # Epochs
    epochs_input = input("Epochs [default 20]: ").strip()
    try:
        epochs = int(epochs_input) if epochs_input else 20
    except:
        epochs = 20
        print(" Using default 20 epochs")
    
    # Initialize components
    mm = ModelManager()
    trainer = ModelTrainer(mm)
    
    # Train or load model
    model, history, loaded = trainer.train_or_load_model(
        dataset_name, model_type, force_retrain, epochs
    )
    
    if model is None:
        print(" Failed to get model. Exiting analysis.")
        return
    
    # Show training history if newly trained
    if not loaded:
        dataset_info = DataProcessor().get_data_summary(dataset_name)
        trainer.plot_training_history(history, model.name, dataset_info["name"])
    
    # Load test data for XAI
    dp = DataProcessor()
    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
    class_names = dp.get_class_names(dataset_name)
    
    # Run comprehensive XAI analysis
    xai = XAIAnalyzer(model, class_names)
    xai.run_complete_xai_analysis(x_test_norm, y_test, dataset_name)
    
    print(f"\n Single model analysis completed for {model_type} on {dataset_name}")

def comparative_analysis():
    """Run comprehensive comparative analysis across all models and datasets"""
    print("\n" + "=" * 60)
    print(" COMPARATIVE MODEL ANALYSIS")
    print("=" * 60)
    
    print("\n[WARN]  This feature is not available in the current version.")
    print("   Please use Option 1 or 7 for model training and analysis.")
    return
    
    mm = ModelManager()
    # cv = ConsolidatedVisualizer(mm)
    
    datasets = ["mnist", "fashion", "cifar10"]
    
    print(" Running comprehensive comparative analysis...")
    print("This may take a few minutes to load all models and generate visualizations.")
    
    # 1. Training history comparison
    cv.plot_consolidated_training_history(datasets)
    
    # 2. Performance comparison
    performance_df = cv.plot_performance_comparison(datasets)
    
    # 3. Model complexity analysis
    cv.plot_model_complexity_analysis()
    
    # 4. Research summary
    summary_df = cv.generate_research_summary(datasets)
    
    print(f"\n Comparative analysis completed!")
    print(f"   Analyzed {len(datasets)} datasets across 3 model architectures")
    print(f"   Generated comprehensive performance metrics and visualizations")

def xai_analysis_pretrained():
    """Run XAI analysis on pre-trained models"""
    print("\n" + "=" * 60)
    print(" XAI ANALYSIS ON PRE-TRAINED MODELS")
    print("=" * 60)
    
    mm = ModelManager()
    saved_models = mm.list_saved_models()
    
    if not saved_models:
        print(" No pre-trained models found. Please train models first.")
        return
    
    print("\n Available pre-trained models:")
    for i, model_file in enumerate(saved_models, 1):
        print(f"{i}. {model_file}")
    
    try:
        choice = int(input(f"\nSelect model (1-{len(saved_models)}): ")) - 1
        selected_model = saved_models[choice]
    except (ValueError, IndexError):
        print(" Invalid selection")
        return
    
    # Parse model file name
    # Format: {model_type}_{dataset}.h5
    model_parts = selected_model.replace('.h5', '').split('_')
    model_type = model_parts[0]
    dataset_name = '_'.join(model_parts[1:])  # Handle datasets with underscores
    
    print(f"\n Running XAI analysis on: {model_type} - {dataset_name}")
    
    # Load model and data
    model = mm.load_model(dataset_name, model_type)
    if model is None:
        print(" Failed to load model")
        return
    
    dp = DataProcessor()
    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
    class_names = dp.get_class_names(dataset_name)
    
    # Run XAI analysis
    xai = XAIAnalyzer(model, class_names)
    xai.run_complete_xai_analysis(x_test_norm, y_test, dataset_name)

def research_summary():
   
    print("\n" + "=" * 60)
    print(" RESEARCH SUMMARY REPORT")
    print("=" * 60)
    
    print("\n[WARN]  This feature is not available in the current version.")
    print("   Please use Option 1 or 7 for model training and analysis.")
    return
    
    mm = ModelManager()
    # cv = ConsolidatedVisualizer(mm)
    
    datasets = ["mnist", "fashion", "cifar10"]
    summary_df = cv.generate_research_summary(datasets)
    
    if summary_df is not None and not summary_df.empty:
        print(f"\n Summary Statistics:")
        print(f"   Total experiments: {len(summary_df)}")
        print(f"   Best accuracy: {summary_df['Accuracy'].max():.4f}")
        print(f"   Average accuracy: {summary_df['Accuracy'].mean():.4f}")
        print(f"   Models analyzed: {summary_df['Model'].nunique()}")
        print(f"   Datasets analyzed: {summary_df['Dataset'].nunique()}")

def list_pretrained_models():
    
    print("\n" + "=" * 60)
    print(" AVAILABLE PRE-TRAINED MODELS")
    print("=" * 60)
    
    mm = ModelManager()
    saved_models = mm.list_saved_models()
    
    if not saved_models:
        print(" No pre-trained models found.")
        print("   Train models using Option 1 from the main menu.")
    else:
        print(f"\n Found {len(saved_models)} pre-trained models:")
        for i, model_file in enumerate(saved_models, 1):
            print(f"   {i}. {model_file}")

def quick_start():
    """Run complete pipeline for quick demonstration"""
    print("\n" + "=" * 60)
    print(" QUICK START - COMPLETE PIPELINE")
    print("=" * 60)
    
    print("\n This will run a complete demonstration with:")
    print("   - MobileNetV2 on Fashion-MNIST")
    print("   - Training (if model doesn't exist)")
    print("   - Comprehensive XAI analysis")
    print("   - Performance evaluation")
    
    confirm = input("\nContinue? (y/n) [default y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Quick start cancelled.")
        return
    
    # Use MobileNetV2 on Fashion-MNIST for quick demonstration
    dataset_name = "fashion"
    model_type = "mobilenet_v2"
    
    mm = ModelManager()
    trainer = ModelTrainer(mm)
    
    print(f"\n Setting up {model_type} on {dataset_name}...")
    
    # Train or load model (quick training with fewer epochs)
    model, history, loaded = trainer.train_or_load_model(
        dataset_name, model_type, force_retrain=False, epochs=10
    )
    
    if model is None:
        print(" Failed to setup model.")
        return
    
    # Load data for analysis
    dp = DataProcessor()
    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
    class_names = dp.get_class_names(dataset_name)
    
    # Quick XAI analysis (limited samples for speed)
    print("\n Running quick XAI analysis...")
    xai = XAIAnalyzer(model, class_names)
    
    # Only run key analyses for quick demo
    accuracy, macro_f1 = xai.evaluate_full_metrics(x_test_norm, y_test)
    xai.grad_cam_analysis(x_test_norm[:3], y_test[:3], dataset_name, num_samples=3)
    
    print(f"\n Quick start completed!")
    print(f"   Model: {model_type}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")

def data_centric_workflow():
    """
    Complete data-centric workflow:
    1. Train baseline model (or load existing)
    2. Identify errors with XAI
    3. Augment error samples
    4. Retrain and compare
    """
    print("\n" + "=" * 70)
    print(" DATA-CENTRIC WORKFLOW: ERROR-DRIVEN AUGMENTATION")
    print("=" * 70)
    print("\nThis workflow demonstrates how XAI-guided data augmentation")
    print("can improve model accuracy through targeted data enhancement.")
    
    # Step 1: Select dataset and model
    print("\n" + "=" * 70)
    print(" STEP 1: CONFIGURATION")
    print("=" * 70)
    
    print("\n[DATA] Select Dataset:")
    print("1. MNIST (Handwritten digits)")
    print("2. Fashion-MNIST (Clothing items)")
    print("3. CIFAR-10 (Color objects)")
    
    ds_choice = get_user_choice("Choice (1-3): ", ["1", "2", "3"])
    ds_map = {"1": "mnist", "2": "fashion", "3": "cifar10"}
    dataset_name = ds_map[ds_choice]
    
    print("\n[BUILD]  Select Model Architecture:")
    print("1. MobileNetV2 (Lightweight, fast)")
    print("2. Efficient CNN (Best accuracy)")
    print("3. ResNet18 (Most stable)")
    
    m_choice = get_user_choice("Choice (1-3): ", ["1", "2", "3"])
    m_map = {"1": "mobilenet_v2", "2": "efficient_cnn", "3": "resnet18"}
    model_type = m_map[m_choice]
    
    print(f"\n[OK] Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model: {model_type}")
    
    # Step 2: Get baseline model
    print("\n" + "=" * 70)
    print(" STEP 2: BASELINE MODEL")
    print("=" * 70)
    
    mm = ModelManager()
    dp = DataProcessor()
    trainer = ModelTrainer(mm)
    
    print("\n[SETUP] Checking for existing baseline model...")
    baseline_model, history, loaded = trainer.train_or_load_model(
        dataset_name, model_type, force_retrain=False, epochs=10
    )
    
    if baseline_model is None:
        print("\n[FAIL] Failed to get baseline model. Exiting workflow.")
        return
    
    # Load data
    print("\n[DIR] Loading dataset...")
    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
    
    # Evaluate baseline
    print("\n[DATA] Evaluating baseline model...")
    baseline_acc, _, _ = trainer.evaluate_model(baseline_model, x_test_norm, y_test)
    print(f"\n[OK] Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    
    # Step 3: Extract and augment errors
    print("\n" + "=" * 70)
    print(" STEP 3: ERROR-DRIVEN AUGMENTATION")
    print("=" * 70)
    
    print("\n[CONFIG]  Augmentation Settings:")
    multiplier_input = input("Augmentation multiplier (recommended 1-3, default 2): ").strip()
    try:
        multiplier = int(multiplier_input) if multiplier_input else 2
        if multiplier < 1:
            print("[WARN]  Multiplier must be at least 1. Using default: 2")
            multiplier = 2
    except:
        multiplier = 2
        print("[WARN]  Invalid input. Using default: 2")
    
    print(f"[OK] Using {multiplier}x augmentation")
    
    x_train_aug, y_train_aug, num_errors, num_augmented = run_phase3_augmentation(
        baseline_model, 
        x_train_norm, 
        y_train, 
        x_test_norm, 
        y_test,
        dataset_name, 
        model_type,
        multiplier=multiplier,
        save=True
    )
    
    if num_errors == 0:
        print("\n[DONE] Perfect baseline accuracy! No errors to augment.")
        print("   The data-centric workflow cannot improve further.")
        return
    
    # Step 4: Retrain, compare, and report
    print("\n" + "=" * 70)
    print(" STEP 4: RETRAIN & VALIDATION")
    print("=" * 70)
    
    print("\n[CONFIG]  Training Settings:")
    epochs_input = input("Training epochs (recommended 5-15, default 10): ").strip()
    try:
        epochs = int(epochs_input) if epochs_input else 10
        if epochs < 1:
            print("[WARN]  Epochs must be at least 1. Using default: 10")
            epochs = 10
    except:
        epochs = 10
        print("[WARN]  Invalid input. Using default: 10")
    
    print(f"[OK] Training for {epochs} epochs")
    
    augmented_model, augmented_acc, improvement = run_phase4_training(
        x_train_aug, y_train_aug, 
        dataset_name, model_type,
        baseline_acc, num_errors, num_augmented,
        len(x_train), epochs=epochs
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print(" WORKFLOW COMPLETE! [DONE]")
    print("=" * 70)
    
    print(f"\n[DATA] Summary:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model: {model_type}")
    print(f"   Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"   Augmented Accuracy: {augmented_acc:.4f} ({augmented_acc*100:.2f}%)")
    print(f"   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        print(f"\n[OK] Data-centric approach validated!")
        print(f"   XAI-guided augmentation improved accuracy by {improvement*100:.2f}%")
    elif improvement == 0:
        print(f"\n✓ Accuracy maintained (dataset may be optimal)")
    else:
        print(f"\n[WARN]  Review augmentation parameters for better results")
    
    print(f"\nFiles saved:")
    print(f"   - Augmented model: saved_models/")
    print(f"   - Augmented dataset: augmented_datasets/")
    print(f"   - Reports: reports/")
    print(f"   - Plots: results/")

def main():
    
    print_banner()
    
    while True:
        try:
            print_menu()
            choice = get_user_choice("\nEnter your choice (1-8): ", 
                                   ["1", "2", "3", "4", "5", "6", "7", "8"])
            
            if choice == "1":
                single_model_analysis()
            elif choice == "2":
                comparative_analysis()
            elif choice == "3":
                xai_analysis_pretrained()
            elif choice == "4":
                research_summary()
            elif choice == "5":
                list_pretrained_models()
            elif choice == "6":
                quick_start()
            elif choice == "7":
                data_centric_workflow()
            elif choice == "8":
                print("\n Exiting......")
                break
            
            # Pause before showing menu again
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n Exiting...")
            break
        except Exception as e:
            print(f"\n An error occurred: {e}")
            print("Please try again or check the error message above.")

if __name__ == "__main__":
    main()


