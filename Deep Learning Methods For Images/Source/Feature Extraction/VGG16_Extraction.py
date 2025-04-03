import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

from .Feature_Extraction import _FeatureExtraction

class VGG16Extraction(_FeatureExtraction):

    def __init__(self,categories):
        super().__init__(categories)
        #Load the VGG16 model
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = torch.nn.Identity()  #Remove the classification layer
        self.model.eval()  #Set the model to evaluation mode
    

    def _preprocess_image(self,img_path):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')  #Convert to RGB

        img = transform(img)
        img = img.unsqueeze(0)  #Add batch dimension
        return img

    def _extract_features(self,img_path):
        img_tensor = self._preprocess_image(img_path)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.numpy().flatten()  #Flatten the features

    def _store_features_and_labels(self,label,data_path,features,labels):

        for image in os.listdir(data_path):

            image_path = os.path.join(data_path,image)
            extracted_features = self._extract_features(image_path)

            features.append(extracted_features)
            labels.append(label)

    def feature_extraction(self):

        for i in self.categories:
            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)

            self._store_features_and_labels(i,train_data_path,self.train_features,self.train_lables)
            self._store_features_and_labels(i,test_data_path,self.test_features,self.test_lables)
