import os
import pickle
import tensorflow as tf

class ModelManager:
    def __init__(self, models_dir="saved_models", history_dir="training_history"):
        self.models_dir = models_dir
        self.history_dir = history_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

    def get_model_path(self, dataset_name, model_type):
        return os.path.join(self.models_dir, f"{model_type}_{dataset_name}.h5")

    def get_history_path(self, dataset_name, model_type):
        return os.path.join(self.history_dir, f"{model_type}_{dataset_name}.pkl")

    def save_model(self, model, dataset_name, model_type):
        path = self.get_model_path(dataset_name, model_type)
        model.save(path)
        print(f"Model saved: {path}")

    def load_model(self, dataset_name, model_type):
        path = self.get_model_path(dataset_name, model_type)
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            print(f" Model loaded: {path}")
            return model
        else:
            print(f" No saved model found: {path}")
            return None

    def save_history(self, history, dataset_name, model_type):
        path = self.get_history_path(dataset_name, model_type)
        with open(path, "wb") as f:
            pickle.dump(history.history, f)
        print(f"Training history saved: {path}")

    def load_history(self, dataset_name, model_type):
        path = self.get_history_path(dataset_name, model_type)
        if os.path.exists(path):
            with open(path, "rb") as f:
                hist = pickle.load(f)
            print(f" Training history loaded: {path}")
            return hist
        else:
            print(f" No saved history found: {path}")
            return None

    def model_exists(self, dataset_name, model_type):
        return os.path.exists(self.get_model_path(dataset_name, model_type))

    def list_saved_models(self):
        """List all saved models"""
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]

