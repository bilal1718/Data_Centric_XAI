"""
Phase 3: Error-Driven Augmentation Orchestrator
Coordinates the complete error analysis and targeted augmentation workflow
"""

from error_extractor import ErrorSampleExtractor
from augmenter import DataAugmenter
from dataset_creator import AugmentedDatasetCreator
from dataset_manager import DatasetManager


def run_phase3_augmentation(model, x_train, y_train, x_test, y_test, 
                            dataset_name, model_type, multiplier=5, save=True):
    """
    Complete Phase 3 workflow: Extract errors → Augment → Create dataset
    
    Args:
        model: Trained baseline model
        x_train: Original training images (normalized)
        y_train: Original training labels
        x_test: Test images (normalized)
        y_test: Test labels
        dataset_name: 'mnist', 'fashion', or 'cifar10'
        model_type: 'mobilenet_v2', 'efficient_cnn', or 'resnet18'
        multiplier: Augmentation multiplier (default 5)
        save: Whether to save augmented dataset (default True)
    
    Returns:
        x_train_augmented: Augmented training set
        y_train_augmented: Augmented training labels
        num_errors: Number of error samples found
        num_augmented: Total augmented samples created
    """
    print("\n" + "=" * 70)
    print(" PHASE 3: ERROR-DRIVEN AUGMENTATION - COMPLETE WORKFLOW")
    print("=" * 70)
    
    # Step 1: Extract error samples
    extractor = ErrorSampleExtractor(model, x_test, y_test)
    error_images, error_labels, error_indices, predictions, confidences = extractor.get_error_samples()
    
    if len(error_images) == 0:
        print("\n[WARN]  No errors to augment. Returning original dataset.")
        return x_train, y_train, 0, 0
    
    # Step 2: Augment error samples
    augmenter = DataAugmenter(dataset_name)
    augmented_images, augmented_labels = augmenter.augment_samples(
        error_images, error_labels, multiplier=multiplier
    )
    
    # Step 3: Create augmented dataset
    creator = AugmentedDatasetCreator()
    x_train_augmented, y_train_augmented = creator.create_augmented_dataset(
        x_train, y_train, augmented_images, augmented_labels
    )
    
    # Step 4: Optionally save
    if save:
        manager = DatasetManager()
        manager.save_augmented_dataset(
            x_train_augmented, y_train_augmented, dataset_name, model_type
        )
    
    print("\n" + "=" * 70)
    print(" PHASE 3 COMPLETE [OK]")
    print("=" * 70)
    print(f"\n[DATA] Final Statistics:")
    print(f"   Error samples identified: {len(error_images)}")
    print(f"   Augmented samples created: {len(augmented_images)}")
    print(f"   Original training size: {len(x_train):,}")
    print(f"   Augmented training size: {len(x_train_augmented):,}")
    print(f"   Dataset growth: +{len(augmented_images):,} samples")
    
    return x_train_augmented, y_train_augmented, len(error_images), len(augmented_images)


