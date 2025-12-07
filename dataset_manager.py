import os
import pickle


class DatasetManager:
    """Save and load augmented datasets for reuse"""
    
    def __init__(self, save_dir="augmented_datasets"):
        """
        Initialize dataset manager
        
        Args:
            save_dir: Directory to save augmented datasets
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_augmented_dataset(self, x_train, y_train, dataset_name, model_type):
        """
        Save augmented dataset for reuse
        
        Args:
            x_train: Augmented training images
            y_train: Augmented training labels
            dataset_name: Name of dataset
            model_type: Model architecture name
        """
        filename = f"{model_type}_{dataset_name}_augmented.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'dataset_name': dataset_name,
            'model_type': model_type,
            'size': len(x_train)
        }
        
        print(f"\n[SAVE] Saving augmented dataset...")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   File: {filepath}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Samples: {len(x_train):,}")
        print(f"[OK] Augmented dataset saved successfully")
    
    def load_augmented_dataset(self, dataset_name, model_type):
        """
        Load previously saved augmented dataset
        
        Args:
            dataset_name: Name of dataset
            model_type: Model architecture name
        
        Returns:
            x_train: Augmented training images (or None)
            y_train: Augmented training labels (or None)
        """
        filename = f"{model_type}_{dataset_name}_augmented.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        if os.path.exists(filepath):
            print(f"\n[DIR] Loading augmented dataset: {filepath}")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            print(f"   Loaded {data['size']:,} samples")
            print(f"[OK] Augmented dataset loaded successfully")
            return data['x_train'], data['y_train']
        else:
            print(f"\n[WARN]  No augmented dataset found: {filepath}")
            return None, None
    
    def list_augmented_datasets(self):
        """List all saved augmented datasets"""
        if not os.path.exists(self.save_dir):
            print(f"\nNo augmented datasets directory found")
            return []
        
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.pkl')]
        
        if not files:
            print(f"\nNo augmented datasets saved yet")
            return []
        
        print(f"\nSaved Augmented Datasets:")
        for i, file in enumerate(files, 1):
            filepath = os.path.join(self.save_dir, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   {i}. {file} ({size_mb:.2f} MB)")
        
        return files


