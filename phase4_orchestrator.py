"""
Phase 4: Data-Centric Training Orchestrator
Coordinates retraining, comparison, and reporting workflow
"""

from augmented_trainer import DataCentricTrainer
from results_comparator import ResultsComparator
from datacentric_report import DataCentricReport


def run_phase4_training(x_train_augmented, y_train_augmented, dataset_name, model_type,
                       baseline_acc, num_errors, num_augmented,
                       original_train_size, epochs=20):
    """
    Complete Phase 4 workflow: Retrain → Compare → Report
    
    Args:
        x_train_augmented: Augmented training images
        y_train_augmented: Augmented training labels
        dataset_name: 'mnist', 'fashion', or 'cifar10'
        model_type: 'mobilenet_v2', 'efficient_cnn', or 'resnet18'
        baseline_acc: Baseline accuracy before augmentation
        num_errors: Number of error samples found
        num_augmented: Total augmented samples created
        original_train_size: Original training set size
        epochs: Training epochs (default 20)
    
    Returns:
        augmented_model: Trained model on augmented data
        augmented_acc: Test accuracy on augmented model
        improvement: Accuracy improvement (augmented - baseline)
    """
    print("\n" + "=" * 70)
    print(" PHASE 4: DATA-CENTRIC TRAINING - COMPLETE WORKFLOW")
    print("=" * 70)
    
    # Step 1: Train on augmented dataset
    trainer = DataCentricTrainer()
    augmented_model, history, augmented_acc = trainer.train_on_augmented_data(
        x_train_augmented, y_train_augmented,
        dataset_name, model_type, epochs=epochs
    )
    
    # Step 2: Compare results
    comparator = ResultsComparator()
    improvement = comparator.compare_accuracies(
        baseline_acc, augmented_acc, dataset_name, model_type
    )
    
    # Step 3: Generate comprehensive report
    reporter = DataCentricReport()
    report = reporter.generate_report(
        dataset_name, model_type,
        baseline_acc, augmented_acc,
        num_errors, num_augmented,
        original_train_size, len(x_train_augmented)
    )
    
    print("\n" + "=" * 70)
    print(" PHASE 4 COMPLETE [OK]")
    print("=" * 70)
    
    return augmented_model, augmented_acc, improvement


