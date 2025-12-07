import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmenter:
    """Apply targeted augmentation to error-prone samples"""
    
    def __init__(self, dataset_name):
        """
        Initialize augmenter with dataset-specific settings
        
        Args:
            dataset_name: 'mnist', 'fashion', or 'cifar10'
        """
        self.dataset_name = dataset_name
        
        print(f"\n[CONFIG]  Configuring augmentation for {dataset_name}...")
        
        # Define augmentation strategy per dataset
        if dataset_name in ["mnist", "fashion"]:
            self.augmenter = ImageDataGenerator(
                rotation_range=15,           # ±15 degrees
                width_shift_range=0.1,       # 10% horizontal shift
                height_shift_range=0.1,      # 10% vertical shift
                shear_range=0.1,            # Shear transformation
                zoom_range=0.1,             # 10% zoom
                fill_mode='nearest'
            )
            print(f"   Grayscale augmentation: rotation, shifts, shear, zoom")
        elif dataset_name == "cifar10":
            self.augmenter = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,        # Flip for CIFAR (cars, animals)
                zoom_range=0.1,
                fill_mode='nearest'
            )
            print(f"   Color augmentation: rotation, shifts, flip, zoom")
    
    def augment_samples(self, error_images, error_labels, multiplier=5):
        """
        Create multiple augmented versions of each error sample
        
        Args:
            error_images: np.array of misclassified images
            error_labels: np.array of true labels
            multiplier: how many augmented versions per error (default 5)
        
        Returns:
            augmented_images: np.array of all augmented versions
            augmented_labels: np.array of corresponding labels
        """
        if len(error_images) == 0:
            print("\n[WARN]  No error samples to augment")
            return np.array([]), np.array([])
        
        print("\n" + "=" * 60)
        print(" AUGMENTING ERROR SAMPLES")
        print("=" * 60)
        print(f"\n[RUN] Creating {multiplier}x augmented versions for each error...")
        
        augmented_images = []
        augmented_labels = []
        
        for i, (img, label) in enumerate(zip(error_images, error_labels)):
            # Add original error sample
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Generate augmented versions
            img_expanded = np.expand_dims(img, 0)  # Add batch dimension
            
            aug_iter = self.augmenter.flow(
                img_expanded, 
                batch_size=1,
                shuffle=False
            )
            
            for _ in range(multiplier):
                augmented = next(aug_iter)[0]
                augmented_images.append(augmented)
                augmented_labels.append(label)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(error_images):
                print(f"   Processed: {i+1}/{len(error_images)} error samples", end='\r')
        
        print()  # New line after progress
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"\n[CHART] Augmentation Summary:")
        print(f"   Original error samples: {len(error_images)}")
        print(f"   Augmentation multiplier: {multiplier}x")
        print(f"   Total augmented samples: {len(augmented_images)}")
        print(f"   New samples created: {len(augmented_images) - len(error_images)}")
        
        return augmented_images, augmented_labels


