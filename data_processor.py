import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.dataset_info = {
            "mnist": {"shape": (32, 32, 1), "classes": 10, "name": "MNIST"},
            "fashion": {"shape": (32, 32, 1), "classes": 10, "name": "Fashion-MNIST"},
            "cifar10": {"shape": (32, 32, 3), "classes": 10, "name": "CIFAR-10"},
        }
        
        self.class_names = {
            "mnist": [str(i) for i in range(10)],
            "fashion": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "cifar10": ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                       "Dog", "Frog", "Horse", "Ship", "Truck"],
        }

    def load_dataset(self, dataset_name):
        """Load and preprocess the specified dataset"""
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {list(self.dataset_info.keys())}")

        print(f" Loading {self.dataset_info[dataset_name]['name']} dataset...")
        
        if dataset_name == "mnist":
            (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
            x_train = self._resize_images(x_train, (32, 32))
            x_test = self._resize_images(x_test, (32, 32))
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            
        elif dataset_name == "fashion":
            (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
            x_train = self._resize_images(x_train, (32, 32))
            x_test = self._resize_images(x_test, (32, 32))
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            
        elif dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            y_train = y_train.flatten()
            y_test = y_test.flatten()

        print(f" Dataset loaded: {self.dataset_info[dataset_name]['name']}")
        print(f"   Training data: {x_train.shape}, {y_train.shape}")
        print(f"   Test data: {x_test.shape}, {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)

    def _resize_images(self, images, target_size):
        """Resize images to target size"""
        resized = []
        for img in images:
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            resized_img = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_AREA)
            resized_img = resized_img.astype("float32") / 255.0
            resized.append(resized_img)
        resized = np.array(resized)
        if resized.ndim == 3:
            resized = np.expand_dims(resized, axis=-1)
        return resized

    def apply_normalization(self, x_train, x_test):
        """Apply per-channel normalization"""
        mean = np.mean(x_train, axis=(0, 1, 2))
        std = np.std(x_train, axis=(0, 1, 2))
        x_train_norm = (x_train - mean) / (std + 1e-7)
        x_test_norm = (x_test - mean) / (std + 1e-7)

        if np.isscalar(mean) or (hasattr(mean, "shape") and np.array(mean).shape == ()):
            print(f" Normalization applied - Mean: {float(mean):.4f}, Std: {float(std):.4f}")
        else:
            mean_str = ", ".join([f"{m:.4f}" for m in np.ravel(mean)])
            std_str = ", ".join([f"{s:.4f}" for s in np.ravel(std)])
            print(f" Normalization applied - Mean: [{mean_str}], Std: [{std_str}]")
            
        return x_train_norm, x_test_norm

    def get_dataset_info(self, dataset_name):
        """Get dataset information"""
        return self.dataset_info.get(dataset_name)

    def get_class_names(self, dataset_name):
        """Get class names for the dataset"""
        return self.class_names.get(dataset_name, [str(i) for i in range(10)])

    def prepare_training_data(self, x_train, y_train, validation_split=0.2, random_state=42):
        """Split data into training and validation sets"""
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=validation_split, random_state=random_state
        )
        print(f" Data split - Training: {x_train_split.shape}, Validation: {x_val.shape}")
        return x_train_split, x_val, y_train_split, y_val

    def get_data_summary(self, dataset_name):
        """Get comprehensive dataset summary"""
        info = self.dataset_info.get(dataset_name, {})
        return {
            "name": info.get("name", dataset_name),
            "shape": info.get("shape", (32, 32, 3)),
            "classes": info.get("classes", 10),
            "class_names": self.get_class_names(dataset_name)
        }

