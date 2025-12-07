import numpy as np


class AugmentedDatasetCreator:
    """Combine original training data with augmented error samples"""
    
    def create_augmented_dataset(self, x_train_original, y_train_original, 
                                 augmented_samples, augmented_labels):
        """
        Merge original training set with augmented error samples
        
        Args:
            x_train_original: Original training images
            y_train_original: Original training labels
            augmented_samples: Augmented error samples
            augmented_labels: Labels for augmented samples
        
        Returns:
            x_train_augmented: Combined dataset
            y_train_augmented: Combined labels
        """
        print("\n" + "=" * 60)
        print(" CREATING AUGMENTED DATASET")
        print("=" * 60)
        
        if len(augmented_samples) == 0:
            print("\n[WARN]  No augmented samples to add, returning original dataset")
            return x_train_original, y_train_original
        
        print(f"\n[DATA] Dataset Composition:")
        print(f"   Original training size: {len(x_train_original):,}")
        print(f"   Augmented samples: {len(augmented_samples):,}")
        
        # Combine datasets
        x_train_augmented = np.concatenate([x_train_original, augmented_samples], axis=0)
        y_train_augmented = np.concatenate([y_train_original, augmented_labels], axis=0)
        
        # Shuffle to mix original and augmented
        print(f"\n[SHUFFLE] Shuffling combined dataset...")
        shuffle_idx = np.random.permutation(len(x_train_augmented))
        x_train_augmented = x_train_augmented[shuffle_idx]
        y_train_augmented = y_train_augmented[shuffle_idx]
        
        increase_pct = len(augmented_samples) / len(x_train_original) * 100
        
        print(f"\n[OK] Augmented Dataset Created:")
        print(f"   Final training set size: {len(x_train_augmented):,}")
        print(f"   Dataset increase: +{len(augmented_samples):,} samples (+{increase_pct:.2f}%)")
        print(f"   Data mix: {len(x_train_original):,} original + {len(augmented_samples):,} augmented")
        
        return x_train_augmented, y_train_augmented


