import tensorflow as tf
from tensorflow.keras import layers, Model

class ModelBuilder:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_mobilenet_v2(self):
  
        print(" Building MobileNetV2 architecture...")
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        def inverted_residual_block(x, filters, stride, expansion):
            shortcut = x
            in_ch = int(x.shape[-1])
            expanded = int(in_ch * expansion)
            
            # Expansion phase
            if expansion != 1:
                x = layers.Conv2D(expanded, (1, 1), padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU(6.0)(x)
            
            # Depthwise convolution
            x = layers.DepthwiseConv2D((3, 3), strides=(stride, stride), padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.0)(x)
            
            # Projection phase
            x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            
            # Residual connection
            if stride == 1 and in_ch == filters:
                x = layers.Add()([x, shortcut])
            return x

        # Stack of inverted residual blocks
        x = inverted_residual_block(x, 16, 1, 1)
        x = inverted_residual_block(x, 24, 2, 6)
        x = inverted_residual_block(x, 24, 1, 6)
        x = inverted_residual_block(x, 32, 2, 6)
        x = inverted_residual_block(x, 32, 1, 6)
        x = inverted_residual_block(x, 64, 2, 6)
        x = inverted_residual_block(x, 64, 1, 6)

        # Final layers
        x = layers.Conv2D(1280, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = Model(inputs, outputs, name="MobileNetV2_Custom")
        print(f" MobileNetV2 built with {model.count_params():,} parameters")
        return model

    def build_efficient_cnn(self):
     
        print(" Building Efficient CNN architecture...")
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Multi-scale feature extraction
        b1 = layers.Conv2D(32, 1, padding="same", activation="relu")(inputs)  # 1x1
        b2 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)  # 3x3
        b3 = layers.Conv2D(32, 5, padding="same", activation="relu")(inputs)  # 5x5
        b4 = layers.MaxPooling2D(3, strides=1, padding="same")(inputs)        # MaxPool
        
        # Concatenate multi-scale features
        x = layers.Concatenate()([b1, b2, b3, b4])
        x = layers.BatchNormalization()(x)

        def depthwise_block(x, filters, stride=1):
          
            x = layers.DepthwiseConv2D(3, strides=stride, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, 1, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x

        # Stack of depthwise blocks
        x = depthwise_block(x, 64)
        x = layers.MaxPooling2D(2)(x)
        x = depthwise_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        x = depthwise_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        x = depthwise_block(x, 512)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = Model(inputs, outputs, name="Efficient_CNN")
        print(f" Efficient CNN built with {model.count_params():,} parameters")
        return model

    def build_resnet18(self):
    
        print(" Building ResNet18 architecture...")
        
        inputs = tf.keras.Input(shape=self.input_shape)

        def residual_block(x, filters, stride=1):
            """Residual block with projection shortcut"""
            shortcut = x
            
            # First convolution
            x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Second convolution
            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            # Projection shortcut if needed
            if stride != 1 or int(shortcut.shape[-1]) != filters:
                shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add residual connection
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            return x

        # Initial layers
        x = layers.Conv2D(64, (3, 3), strides=1, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = Model(inputs, outputs, name="ResNet18_Custom")
        print(f" ResNet18 built with {model.count_params():,} parameters")
        return model

    def build_model(self, model_type):
       
        model_builders = {
            "mobilenet_v2": self.build_mobilenet_v2,
            "efficient_cnn": self.build_efficient_cnn,
            "resnet18": self.build_resnet18
        }
        
        if model_type not in model_builders:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_builders.keys())}")
        
        return model_builders[model_type]()

    def get_model_info(self):
        
        return {
            "mobilenet_v2": "Lightweight architecture with depthwise convolutions",
            "efficient_cnn": "Multi-scale feature extraction with depthwise blocks", 
            "resnet18": "Residual learning with identity mappings"
        }

