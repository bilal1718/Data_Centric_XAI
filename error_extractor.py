import numpy as np


class ErrorSampleExtractor:
    """Extract and organize misclassified samples from XAI analysis"""
    
    def __init__(self, model, x_test, y_test):
        """
        Initialize error sample extractor
        
        Args:
            model: Trained Keras model
            x_test: Test images (normalized)
            y_test: True test labels
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
    
    def get_error_samples(self):
        """
        Extract all misclassified samples
        
        Returns:
            error_images: np.array of misclassified images
            error_labels: np.array of true labels
            error_indices: indices in original test set
            predictions: what model predicted (wrong)
            confidences: confidence scores of wrong predictions
        """
        print("\n" + "=" * 60)
        print(" EXTRACTING ERROR SAMPLES")
        print("=" * 60)
        
        # Get predictions
        y_pred_probs = self.model.predict(self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Find misclassifications
        mis_idx = np.where(y_pred != self.y_test)[0]
        
        if len(mis_idx) == 0:
            print("\n[DONE] Perfect accuracy! No misclassified samples found.")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # Extract error data
        error_images = self.x_test[mis_idx]
        error_labels = self.y_test[mis_idx]
        predictions = y_pred[mis_idx]
        confidences = np.max(y_pred_probs[mis_idx], axis=1)
        
        print(f"\n[DATA] Error Analysis:")
        print(f"   Total test samples: {len(self.y_test)}")
        print(f"   Misclassified: {len(mis_idx)}")
        print(f"   Error rate: {len(mis_idx)/len(self.y_test)*100:.2f}%")
        print(f"   Accuracy: {(1 - len(mis_idx)/len(self.y_test))*100:.2f}%")
        
        # Show top confusion pairs
        unique_errors = {}
        for true_label, pred_label in zip(error_labels, predictions):
            key = (int(true_label), int(pred_label))
            unique_errors[key] = unique_errors.get(key, 0) + 1
        
        if unique_errors:
            print(f"\n   Top confusion pairs:")
            sorted_errors = sorted(unique_errors.items(), key=lambda x: x[1], reverse=True)[:5]
            for (true_lbl, pred_lbl), count in sorted_errors:
                print(f"      True: {true_lbl} → Predicted: {pred_lbl} ({count} samples)")
        
        print(f"\n[OK] Extracted {len(error_images)} error samples for augmentation")
        
        return error_images, error_labels, mis_idx, predictions, confidences


