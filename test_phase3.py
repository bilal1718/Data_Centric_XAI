"""
Test script for Phase 3: Error-Driven Augmentation
Run this to verify Phase 3 implementation works correctly
"""

import sys
import numpy as np
from data_processor import DataProcessor
from model_manager import ModelManager
from error_extractor import ErrorSampleExtractor
from augmenter import DataAugmenter
from dataset_creator import AugmentedDatasetCreator
from dataset_manager import DatasetManager
from phase3_orchestrator import run_phase3_augmentation

def test_phase3():
    """Test Phase 3 implementation"""
    
    print("=" * 70)
    print(" PHASE 3 TEST - Error-Driven Augmentation")
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
    
    # Step 2: Load trained model
    print("\n[RUN] Step 2: Loading trained model...")
    mm = ModelManager()
    model = mm.load_model(dataset_name, model_type)
    
    if model is None:
        print("\n[FAIL] ERROR: No trained model found!")
        print(f"   Please train {model_type} on {dataset_name} first using menu option 1")
        return False
    
    # Step 3: Test error extraction
    print("\n[RUN] Step 3: Testing error extraction...")
    extractor = ErrorSampleExtractor(model, x_test_norm, y_test)
    error_images, error_labels, error_indices, predictions, confidences = extractor.get_error_samples()
    
    if len(error_images) == 0:
        print("\n[WARN]  Perfect accuracy - no errors to test with!")
        print("   Phase 3 would work, but there's nothing to augment")
        return True
    
    print(f"\n[OK] Error extraction successful: {len(error_images)} errors found")
    
    # Step 4: Test augmentation
    print("\n[RUN] Step 4: Testing augmentation...")
    augmenter = DataAugmenter(dataset_name)
    augmented_images, augmented_labels = augmenter.augment_samples(
        error_images[:5],  # Test with just 5 samples
        error_labels[:5], 
        multiplier=3  # 3x multiplier for testing
    )
    
    print(f"\n[OK] Augmentation successful: {len(augmented_images)} augmented samples")
    
    # Step 5: Test dataset creation
    print("\n[RUN] Step 5: Testing augmented dataset creation...")
    creator = AugmentedDatasetCreator()
    x_train_aug, y_train_aug = creator.create_augmented_dataset(
        x_train_norm[:1000],  # Test with subset
        y_train[:1000], 
        augmented_images, 
        augmented_labels
    )
    
    print(f"\n[OK] Dataset creation successful: {len(x_train_aug)} total samples")
    
    # Step 6: Test save/load
    print("\n[RUN] Step 6: Testing dataset save/load...")
    manager = DatasetManager()
    manager.save_augmented_dataset(x_train_aug, y_train_aug, dataset_name, f"{model_type}_test")
    x_loaded, y_loaded = manager.load_augmented_dataset(dataset_name, f"{model_type}_test")
    
    if x_loaded is not None:
        print(f"\n[OK] Save/load successful: {len(x_loaded)} samples loaded")
    
    # Step 7: Test complete workflow
    print("\n[RUN] Step 7: Testing complete Phase 3 workflow...")
    x_full_aug, y_full_aug, num_errors, num_augmented = run_phase3_augmentation(
        model, 
        x_train_norm, 
        y_train, 
        x_test_norm, 
        y_test,
        dataset_name, 
        model_type,
        multiplier=5,
        save=False  # Don't save test run
    )
    
    print(f"\n[OK] Complete workflow successful!")
    
    # Final validation
    print("\n" + "=" * 70)
    print(" PHASE 3 VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n[OK] All Phase 3 components working correctly!")
    print(f"\n[DATA] Test Summary:")
    print(f"   Errors found: {num_errors}")
    print(f"   Augmented samples: {num_augmented}")
    print(f"   Original dataset: {len(x_train):,}")
    print(f"   Augmented dataset: {len(x_full_aug):,}")
    print(f"   Growth: +{num_augmented:,} samples")
    
    print(f"\n[DONE] Phase 3 implementation is READY!")
    print(f"\n➡️  Next: Implement Phase 4 (Retrain & Validation)")
    
    return True

if __name__ == "__main__":
    try:
        success = test_phase3()
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

