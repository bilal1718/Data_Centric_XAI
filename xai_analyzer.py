import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
import logging

# Optional packages (LIME / SHAP)
HAS_LIME = False
HAS_SHAP = False

try:
    from lime import lime_image
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class XAIAnalyzer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        print(f" XAI Analyzer initialized with {len(class_names)} classes")

    def evaluate_full_metrics(self, x_test, y_test):
       
        print("\n" + "=" * 70)
        print(" COMPREHENSIVE PERFORMANCE METRICS")
        print("=" * 70)
        
        # Predictions
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f" Overall Accuracy: {accuracy:.4f}")

        # Detailed classification report
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        print("\n Per-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 65)
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")

        # Confusion matrix visualization
        self._plot_confusion_matrix(y_test, y_pred)
        
        macro_f1 = np.mean(f1)
        print(f"\n Macro F1 Score: {macro_f1:.4f}")
        return accuracy, macro_f1

    def _plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix with professional styling"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'shrink': 0.8})
        
        plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
        plt.ylabel("True Label", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def grad_cam_analysis(self, images, true_labels, dataset_name, num_samples=5):
    
        print("\n" + "=" * 60)
        print(" GRAD-CAM ANALYSIS")
        print("=" * 60)
        
        # Find last convolutional layer
        last_conv = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, 
                                tf.keras.layers.SeparableConv2D, 
                                tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break
                
        if last_conv is None:
            print(" No convolutional layer found. Skipping Grad-CAM.")
            return

        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(last_conv).output, self.model.output]
        )

        n = min(num_samples, len(images))
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            img = images[i]
            true_label = true_labels[i]
            img_batch = np.expand_dims(img, axis=0)
            
            # Get prediction
            preds = self.model.predict(img_batch, verbose=0)
            pred_label = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))

            try:
                # Grad-CAM computation
                with tf.GradientTape() as tape:
                    conv_outs, predictions = grad_model(img_batch)
                    loss = predictions[:, pred_label]
                
                grads = tape.gradient(loss, conv_outs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outs[0]
                
                # Create heatmap
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                heatmap = np.maximum(heatmap, 0)
                if np.max(heatmap) != 0:
                    heatmap /= np.max(heatmap)
                
                heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
                heatmap_uint8 = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                # Plot original image
                if dataset_name in ["mnist", "fashion"]:
                    axes[0, i].imshow(img.squeeze(), cmap="gray")
                else:
                    axes[0, i].imshow(img)
                
                axes[0, i].set_title(
                    f"True: {self.class_names[int(true_label)]}\n"
                    f"Pred: {self.class_names[int(pred_label)]}",
                    fontsize=10
                )
                axes[0, i].axis("off")

                # Plot Grad-CAM overlay
                if dataset_name in ["mnist", "fashion"]:
                    img_rgb = cv2.cvtColor((img.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = (img * 255).astype(np.uint8)
                
                superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
                axes[1, i].imshow(superimposed)
                axes[1, i].set_title(f"Grad-CAM (Conf: {confidence:.3f})", fontsize=10)
                axes[1, i].axis("off")

            except Exception as e:
                print(f" Grad-CAM failed for sample {i}: {e}")
                # Fallback display
                if dataset_name in ["mnist", "fashion"]:
                    axes[0, i].imshow(img.squeeze(), cmap="gray")
                    axes[1, i].imshow(img.squeeze(), cmap="gray")
                else:
                    axes[0, i].imshow(img)
                    axes[1, i].imshow(img)
                axes[0, i].axis("off")
                axes[1, i].axis("off")

        plt.suptitle("Grad-CAM: Model Attention Visualizations", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def misclassification_analysis(self, x_test, y_test, max_examples=6):
        
        print("\n" + "=" * 60)
        print(" MISCLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        # Get predictions
        y_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)
        
        # Find misclassifications
        mis_idx = np.where(y_pred != y_test)[0]
        if len(mis_idx) == 0:
            print(" No misclassifications found!")
            return
            
        # Select samples to display
        sel = mis_idx if len(mis_idx) <= max_examples else np.random.choice(mis_idx, size=max_examples, replace=False)
        
        # Create visualization
        cols = 2
        rows = int(np.ceil(len(sel) / cols))
        fig, axes = plt.subplots(rows, cols * 2, figsize=(15, rows * 3))
        axes = np.array(axes).reshape(rows, cols * 2)
        
        for i, idx in enumerate(sel):
            img = x_test[idx]
            true = int(y_test[idx])
            pred = int(y_pred[idx])
            prob = float(y_probs[idx][pred])
            
            row = i // cols
            col = (i % cols) * 2
            
            # Plot image
            if img.ndim == 3 and img.shape[2] == 1:
                axes[row, col].imshow(img.squeeze(), cmap="gray")
            else:
                axes[row, col].imshow(img)
                
            axes[row, col].set_title(
                f"True: {self.class_names[true]}\nPred: {self.class_names[pred]}",
                color='red', fontweight='bold'
            )
            axes[row, col].axis("off")
            
            # Plot probability distribution
            bars = axes[row, col + 1].bar(
                range(len(y_probs[idx])), 
                y_probs[idx],
                color=["red" if j == pred else "orange" if j == true else "blue" 
                      for j in range(len(y_probs[idx]))]
            )
            
            axes[row, col + 1].set_xticks(range(len(y_probs[idx])))
            axes[row, col + 1].set_xticklabels(self.class_names, rotation=45, fontsize=8)
            axes[row, col + 1].set_title(f"Confidence: {prob:.3f}", fontweight='bold')
            axes[row, col + 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, y_probs[idx]):
                if val > 0.1:
                    axes[row, col + 1].text(
                        bar.get_x() + bar.get_width() / 2, 
                        bar.get_height() + 0.01, 
                        f"{val:.2f}", 
                        ha="center", va="bottom", fontsize=7
                    )

        plt.suptitle("Misclassification Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print confusion summary
        self._print_confusion_summary(y_test, y_pred)

    def _print_confusion_summary(self, y_test, y_pred):
        """Print summary of confusion patterns"""
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n Confusion Matrix Summary:")
        print(f"   Total test samples: {len(y_test)}")
        
        correct = int(np.sum(y_pred == y_test))
        wrong = int(np.sum(y_pred != y_test))
        
        print(f"   Correct predictions: {correct} ({correct/len(y_test):.2%})")
        print(f"   Misclassifications: {wrong} ({wrong/len(y_test):.2%})")
        
        # Find most confusing pairs
        cm2 = cm.copy()
        np.fill_diagonal(cm2, 0)
        top_indices = np.argsort(cm2.flatten())[-5:][::-1]
        
        print(f"\n Top confusing pairs (True → Predicted):")
        for idx in top_indices:
            if cm2.flatten()[idx] > 0:
                true_class, pred_class = np.unravel_index(idx, cm2.shape)
                count = cm2[true_class, pred_class]
                print(f"   {self.class_names[true_class]} → {self.class_names[pred_class]}: {count} samples")

    def run_complete_xai_analysis(self, x_test, y_test, dataset_name):
     
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE XAI ANALYSIS STARTED")
        print("=" * 80)
        
        try:
            # 1. Full metrics evaluation
            self.evaluate_full_metrics(x_test, y_test)
        except Exception as e:
            print(f" Full metrics failed: {e}")
        
        try:
            # 2. Grad-CAM analysis
            self.grad_cam_analysis(x_test[:5], y_test[:5], dataset_name, num_samples=5)
        except Exception as e:
            print(f" Grad-CAM failed: {e}")
        
        try:
            # 3. Misclassification analysis
            self.misclassification_analysis(x_test, y_test, max_examples=6)
        except Exception as e:
            print(f" Misclassification analysis failed: {e}")
        
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE XAI ANALYSIS COMPLETED")
        print("=" * 80)

