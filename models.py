#%
import torch
import torch.nn as nn
import torchvision, torchinfo, torchmetrics
import torchvision

# MODEL ARCHITECTURE
reduce_by_half = nn.MaxPool2d ((2,2), 2)
feature_extractor = nn.Sequential(
        # Standard Input Size: 3, 224, 224
        # Halving needed to reach (7,7) = 224 -> 112 -> 56 -> 28 -> 14 -> 7
        nn.Conv2d ( in_channels =  3, out_channels = 50, kernel_size = (3,3), padding="same"), nn.ReLU(),
        reduce_by_half,
        # 112, 112
        nn.Conv2d ( in_channels = 50, out_channels = 100, kernel_size = (3,3), padding="same"), nn.ReLU(), 
        reduce_by_half,
        # 56, 56
        nn.Conv2d ( in_channels = 100, out_channels = 200, kernel_size = (3,3), padding="same"), nn.ReLU(),
        reduce_by_half,
        # 28, 28
        nn.Conv2d ( in_channels = 200, out_channels = 300, kernel_size = (3,3), padding="same"), nn.ReLU(),
        reduce_by_half,
        # 14, 14
        nn.Conv2d ( in_channels = 300, out_channels = 512, kernel_size = (3,3), padding="same"), nn.ReLU(), 
        reduce_by_half,
        # Standard Output Size: 512, 7, 7
        # Total Features = 512 images of 7*7 pixel
)

decision_maker = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features = 512*7*7 , out_features = 50), nn.ReLU(),
        nn.Linear(in_features = 50      , out_features = 2) # 1 Neuron is for detecting Cat and 2nd Neuron is for detecting Dog
)

model = nn.Sequential(
    feature_extractor,
    decision_maker
)

#%
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
keras.config.set_image_data_format('channels_first')

# Filter: UV Light filter on glasses is looking for UV lights to filter out or Filter in water purification, is looking for impurities
# Halving needed to reach (7,7) = 224 -> 112 -> 56 -> 28 -> 14 -> 7
reduce_by_half = keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2)
feature_extractor = keras.Sequential([
    # input: (3, 224, 224)
    keras.layers.Input(shape = (3, 224, 224)),
    keras.layers.Conv2D( filters = 50, kernel_size = (3,3), activation="relu" , padding="same"),
    reduce_by_half,
    keras.layers.Conv2D( filters = 100, kernel_size = (3,3), activation="relu", padding="same"),
    reduce_by_half,
    keras.layers.Conv2D( filters = 200, kernel_size = (3,3), activation="relu", padding="same"),
    reduce_by_half,
    keras.layers.Conv2D( filters = 300, kernel_size = (3,3), activation="relu", padding="same"),
    reduce_by_half,
    keras.layers.Conv2D( filters = 400, kernel_size = (3,3), activation="relu", padding="same"),
    reduce_by_half,
    # output: 400, 7, 7
])

decision_maker = keras.Sequential([
    keras.layers.Input((400,7,7)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(units = 100, activation = "relu"),
    keras.layers.Dense(units = 2)
])

model = keras.Sequential([
    feature_extractor,
    decision_maker
])

x = torch.randn((1,3,224,224))
print(model(x).shape)

model.summary()
model.summary(expand_nested= True)

print("hi")
print("hello")

