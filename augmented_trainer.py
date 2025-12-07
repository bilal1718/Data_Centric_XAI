import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from model_trainer import ModelTrainer
from model_manager import ModelManager
from data_processor import DataProcessor
from model_builder import ModelBuilder


class DataCentricTrainer:
    """Orchestrate the complete data-centric training workflow"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.data_processor = DataProcessor()
    
    def train_on_augmented_data(self, x_train_augmented, y_train_augmented, 
                                 dataset_name, model_type, epochs=20):
        """
        Train model on augmented dataset
        
        Args:
            x_train_augmented: Augmented training images
            y_train_augmented: Augmented training labels
            dataset_name: 'mnist', 'fashion', or 'cifar10'
            model_type: 'mobilenet_v2', 'efficient_cnn', or 'resnet18'
            epochs: Training epochs
        
        Returns:
            model: Trained model
            history: Training history
            test_accuracy: Final test accuracy
        """
        print("=" * 70)
        print(" PHASE 4: TRAINING ON AUGMENTED DATASET")
        print("=" * 70)
        
        # Augmented data is ALREADY normalized (passed through from Phase 3)
        print(f"\n[OK] Augmented dataset is already normalized from Phase 3")
        print(f"   Skipping normalization to avoid double-normalization bug")
        
        # Use augmented data as-is (already normalized)
        x_train_norm = x_train_augmented
        
        # Split train/validation
        print(f"[SETUP] Splitting into train/validation...")
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train_norm, y_train_augmented, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"   Training samples: {len(x_train_split):,}")
        print(f"   Validation samples: {len(x_val):,}")
        
        # Build model
        print(f"\n[BUILD]  Building {model_type} architecture...")
        input_shape = self.data_processor.get_dataset_info(dataset_name)["shape"]
        num_classes = self.data_processor.get_dataset_info(dataset_name)["classes"]
        
        model_builder = ModelBuilder(input_shape, num_classes)
        model = model_builder.build_model(model_type)
        
        # Train
        print(f"\n[START] Training on augmented dataset ({epochs} epochs)...")
        trainer = ModelTrainer(self.model_manager)
        history = trainer.train_model(
            model, x_train_split, y_train_split, x_val, y_val, 
            epochs=epochs
        )
        
        # Evaluate on original test set (normalized the same way)
        print(f"\n[DATA] Evaluating on test set...")
        (x_train_orig, _), (x_test, y_test) = self.data_processor.load_dataset(dataset_name)
        
        # Normalize test data using ORIGINAL training stats (same as baseline)
        mean = np.mean(x_train_orig, axis=(0, 1, 2))
        std = np.std(x_train_orig, axis=(0, 1, 2))
        x_test_norm = (x_test - mean) / (std + 1e-7)
        
        test_acc, _, _ = trainer.evaluate_model(model, x_test_norm, y_test)
        
        # Save augmented model with special naming
        print(f"\n[SAVE] Saving augmented model...")
        save_path = self.model_manager.get_model_path(f"{dataset_name}_augmented", model_type)
        model.save(save_path)
        print(f"   Saved: {save_path}")
        
        # Save history
        history_path = self.model_manager.get_history_path(f"{dataset_name}_augmented", model_type)
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"   History saved: {history_path}")
        
        print(f"\n[OK] Augmented model test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        return model, history, test_acc


