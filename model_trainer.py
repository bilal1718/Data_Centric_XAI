import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def train_model(self, model, x_train, y_train, x_val, y_val, epochs=30, batch_size=32):
        
        print(f" Starting training for {epochs} epochs...")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
            ),
        ]
        
        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print(" Training completed!")
        return history

    def evaluate_model(self, model, x_test, y_test, class_names=None):
     
        print("\n Evaluating model performance...")
        
        # Basic evaluation
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f" Test Accuracy: {test_acc:.4f}")
        print(f" Test Loss: {test_loss:.4f}")
        
        # Predictions for detailed metrics
        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f" Sklearn Accuracy: {accuracy:.4f}")
        
        # Classification report
        if class_names:
            print("\n Classification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        
        return test_acc, y_pred_probs, y_pred

    def plot_training_history(self, history, model_name, dataset_name):
       
        if history is None:
            print(" No history to plot.")
            return
            
        # Extract history (handles both History object and dictionary)
        if isinstance(history, dict):
            hist = history
        else:
            hist = history.history
            
        train_acc = hist.get("accuracy", [])
        val_acc = hist.get("val_accuracy", [])
        train_loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(train_acc, label="Training Accuracy", linewidth=2, color='blue')
        ax1.plot(val_acc, label="Validation Accuracy", linewidth=2, color='red')
        ax1.set_title(f"{model_name} - Accuracy\n({dataset_name})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(train_loss, label="Training Loss", linewidth=2, color='blue')
        ax2.plot(val_loss, label="Validation Loss", linewidth=2, color='red')
        ax2.set_title(f"{model_name} - Loss\n({dataset_name})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def calculate_model_complexity(self, model):
     
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print(f" Model Complexity:")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {non_trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        
        return total_params

    def train_or_load_model(self, dataset_name, model_type, force_retrain=False, epochs=30):
       
        # Check if model exists
        if not force_retrain and self.model_manager.model_exists(dataset_name, model_type):
            print(f" Loading existing {model_type} for {dataset_name}...")
            model = self.model_manager.load_model(dataset_name, model_type)
            history = self.model_manager.load_history(dataset_name, model_type)
            return model, history, True
        
        # Train new model
        print(f" Training new {model_type} on {dataset_name}...")
        
        # Load and prepare data
        from data_processor import DataProcessor
        dp = DataProcessor()
        (x_train, y_train), (x_test, y_test) = dp.load_dataset(dataset_name)
        x_train_norm, x_test_norm = dp.apply_normalization(x_train, x_test)
        x_train_split, x_val, y_train_split, y_val = dp.prepare_training_data(x_train_norm, y_train)
        
        # Build model
        input_shape = dp.get_dataset_info(dataset_name)["shape"]
        num_classes = dp.get_dataset_info(dataset_name)["classes"]
        
        from model_builder import ModelBuilder
        mb = ModelBuilder(input_shape, num_classes)
        model = mb.build_model(model_type)
        
        # Train model
        history = self.train_model(model, x_train_split, y_train_split, x_val, y_val, epochs=epochs)
        
        # Evaluate and save
        test_acc, _, _ = self.evaluate_model(model, x_test_norm, y_test)
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        self.model_manager.save_model(model, dataset_name, model_type)
        self.model_manager.save_history(history, dataset_name, model_type)
        
        return model, history, False

