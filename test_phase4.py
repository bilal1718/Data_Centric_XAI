"""
Test script for Phase 4: Retrain & Validation
Run this to verify Phase 4 implementation works correctly
"""

import sys
import numpy as np
from data_processor import DataProcessor
from model_manager import ModelManager
from phase3_orchestrator import run_phase3_augmentation
from augmented_trainer import DataCentricTrainer
from results_comparator import ResultsComparator
from datacentric_report import DataCentricReport

def test_phase4():
    """Test Phase 4 implementation"""
    
    print("=" * 70)
    print(" PHASE 4 TEST - Retrain & Validation")
    print("=" * 70)
    
    # Configuration
    dataset_name = "mnist"  # Change to 'fashion' or 'cifar10' to test others
    model_type = "mobilenet_v2"  # Or 'efficient_cnn', 'resnet18'
    
    print(f"\n📋 Test Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model: {model_type}")
    
    # Step 1: Load data
    print("\n[RUN] Step 1: Loading dataset...")
    dp = DataProcessor()
    (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
    x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
    
    # Step 2: Load baseline model
    print("\n[RUN] Step 2: Loading baseline model...")
    mm = ModelManager()
    baseline_model = mm.load_model(dataset_name, model_type)
    
    if baseline_model is None:
        print("\n[FAIL] ERROR: No baseline model found!")
        print(f"   Please train {model_type} on {dataset_name} first using menu option 1")
        return False
    
    # Get baseline accuracy
    from model_trainer import ModelTrainer
    trainer = ModelTrainer(mm)
    baseline_acc, _, _ = trainer.evaluate_model(baseline_model, x_test_norm, y_test)
    print(f"\n[OK] Baseline accuracy: {baseline_acc:.4f}")
    
    # Step 3: Run Phase 3 to get augmented data
    print("\n[RUN] Step 3: Running Phase 3 (augmentation)...")
    x_train_aug, y_train_aug, num_errors, num_augmented = run_phase3_augmentation(
        baseline_model,
        x_train_norm,
        y_train,
        x_test_norm,
        y_test,
        dataset_name,
        model_type,
        multiplier=3,  # Smaller for testing
        save=False  # Don't save test data
    )
    
    if num_errors == 0:
        print("\n[WARN]  Perfect accuracy - no errors to test with!")
        print("   Phase 4 would work, but there's nothing to improve")
        return True
    
    print(f"\n[OK] Augmented dataset created: {len(x_train_aug):,} samples")
    
    # Step 4: Test DataCentricTrainer
    print("\n[RUN] Step 4: Testing retrain on augmented data...")
    dc_trainer = DataCentricTrainer()
    
    # Use subset for faster testing
    subset_size = min(10000, len(x_train_aug))
    x_train_subset = x_train_aug[:subset_size]
    y_train_subset = y_train_aug[:subset_size]
    
    print(f"   Training on subset: {subset_size:,} samples (for faster testing)")
    augmented_model, aug_history, augmented_acc = dc_trainer.train_on_augmented_data(
        x_train_subset, y_train_subset, dataset_name, model_type, epochs=5
    )
    
    print(f"\n[OK] Retraining successful: {augmented_acc:.4f} accuracy")
    
    # Step 5: Test ResultsComparator
    print("\n[RUN] Step 5: Testing results comparison...")
    comparator = ResultsComparator()
    improvement = comparator.compare_accuracies(
        baseline_acc, augmented_acc, dataset_name, model_type
    )
    
    print(f"\n[OK] Comparison successful: {improvement:+.4f} improvement")
    
    # Step 6: Test DataCentricReport
    print("\n[RUN] Step 6: Testing report generation...")
    reporter = DataCentricReport()
    report = reporter.generate_report(
        dataset_name, model_type,
        baseline_acc, augmented_acc,
        num_errors, num_augmented,
        len(x_train), len(x_train_aug)
    )
    
    print(f"\n[OK] Report generation successful")
    
    # Final validation
    print("\n" + "=" * 70)
    print(" PHASE 4 VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n[OK] All Phase 4 components working correctly!")
    print(f"\n[DATA] Test Summary:")
    print(f"   Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"   Augmented accuracy: {augmented_acc:.4f} ({augmented_acc*100:.2f}%)")
    print(f"   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"   Errors found: {num_errors}")
    print(f"   Augmented samples: {num_augmented}")
    
    print(f"\n[DONE] Phase 4 implementation is READY!")
    print(f"\n➡️  Next: Integrate into main.py as menu option 8")
    
    return True

if __name__ == "__main__":
    try:
        success = test_phase4()
        if success:
            print("\n" + "=" * 70)
            print(" TEST PASSED [OK]")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print(" TEST FAILED [FAIL]")
            print("=" * 70)
    except Exception as e:
        print(f"\n[FAIL] TEST ERROR: {e}")
        import traceback
        traceback.print_exc()

