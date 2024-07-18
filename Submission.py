# DATASET
import torch
import torch.nn as nn
import torchvision, torchinfo, torchmetrics
import torchvision
from sklearn.model_selection import train_test_split
import os, glob, zipfile
from tqdm import tqdm
from PIL import Image

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self,file_list):
        self.file_list = file_list
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)), # IMPORTANT SIZE OF IMAGE (224, 224) or (256,256)
                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self,idx):
        img_path        = self.file_list[idx]
        img             = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0

        return img_transformed,label
    
train_file_names_list = glob.glob(os.path.join("../working/train",'*.jpg'))
test_file_names_list  = glob.glob(os.path.join("../working/test", '*.jpg'))

train_list, val_list  = train_test_split(train_file_names_list, test_size=0.2)

train_dataset           = Custom_Dataset(train_list)
val_dataset             = Custom_Dataset(val_list)

training_dataloader     = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=True )
validation_dataloader   = torch.utils.data.DataLoader(dataset = val_dataset  , batch_size=32, shuffle=False)

# TRAINING
import lightning
import torchmetrics

class Lightning_Module(lightning.LightningModule):
    def __init__(self, model, number_of_classes):
        super().__init__()
        
        self.model = model
        self.automatic_optimization = False
        self.training_accuracy   = torchmetrics.Accuracy(task="multiclass", num_classes=number_of_classes)
        self.validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=number_of_classes)
        
    def training_step (self, batch, batch_idx):
        images_actual, labels_actual        = batch
        predicted_logits  = self.model(images_actual)
        labels_predicted  = torch.argmax(predicted_logits, dim = 1)
        
        loss              = torch.nn.functional.cross_entropy(predicted_logits, labels_actual)
        
        optimizer         = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        """
        for individual_parameter in self.parameters():
            individual_parameter = individual_parameter - individual_parameter.grad * learning_rate
        """
        self.training_accuracy(labels_predicted, labels_actual)
        self.log("train_loss"     , loss                   , prog_bar = True)
        self.log("train_accuracy" , self.training_accuracy , prog_bar = True)
        
        return loss
    
    def validation_step (self, batch, batch_idx):
        images_actual, labels_actual        = batch
        predicted_logits  = self.model(images_actual)
        labels_predicted  = torch.argmax(predicted_logits, dim = 1)
        
        loss = torch.nn.functional.cross_entropy(predicted_logits, labels_actual)
        self.validation_accuracy(labels_predicted, labels_actual)
        self.log("validation_loss"     , loss                     , prog_bar= False)
        self.log("validation_accuracy" , self.validation_accuracy , prog_bar= True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

# MODEL ARCHITECTURE
feature_extractor = nn.Sequential(
        # Standard Input Size: 3, 224, 224
        # Filter: UV Light filter on glasses is looking for UV lights to filter out or Filter in water purification, is looking for impurities
        nn.Conv2d                           ( in_channels =  3, out_channels = 50, kernel_size = (3,3), padding="same"), 
        nn.ReLU(), nn.MaxPool2d ((2,2), 2),
        # 112, 112
        nn.Conv2d                           ( in_channels = 50, out_channels = 50, kernel_size = (3,3), padding="same"),
        nn.ReLU(), nn.MaxPool2d ((2,2), 2),
        # 56, 56
        nn.Conv2d                           ( in_channels = 50, out_channels = 50, kernel_size = (3,3), padding="same"),
        nn.ReLU(), nn.MaxPool2d ((2,2), 2),
        # 28, 28
        nn.Conv2d                           ( in_channels = 50, out_channels = 50, kernel_size = (3,3), padding="same"),
        nn.ReLU(), nn.MaxPool2d ((2,2), 2),
        # 14, 14
        nn.Conv2d                           ( in_channels = 50, out_channels = 512, kernel_size = (3,3), padding="same"),
        nn.ReLU(), nn.MaxPool2d ((2,2), 2),
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

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
keras.config.set_image_data_format('channels_first')

# Halving needed to reach (7,7) = 224 -> 112 -> 56 -> 28 -> 14 -> 7

model = keras.Sequential([
    # input: (3, 224, 224)
    keras.layers.Input(shape = (3, 224, 224)),
    keras.layers.Conv2D( filters = 50, kernel_size = (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2),
    keras.layers.Conv2D( filters = 100, kernel_size = (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2),
    keras.layers.Conv2D( filters = 200, kernel_size = (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2),
    keras.layers.Conv2D( filters = 300, kernel_size = (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2),
    keras.layers.Conv2D( filters = 512, kernel_size = (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D( pool_size = (2,2), strides = 2),
    # output: 512, 7, 7
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(units = 100, activation = "relu"),
    keras.layers.Dense(units = 2)
    
])

epochs  = 10
lightning_model   = Lightning_Module(model, number_of_classes= 2)
lightning_trainer = lightning.Trainer( max_epochs= epochs, log_every_n_steps= 50)

lightning_trainer.fit(model=lightning_model, train_dataloaders= training_dataloader, val_dataloaders= validation_dataloader)