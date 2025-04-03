import os
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image

# Private class for reusing feature extraction in different machine learning or deep learning algorithms and also a parent class for all the other feature extracting classes
class _FeatureExtraction:
    
    def __init__(self,categories):
        self.train_dir = '../../dataset/train'
        self.test_dir = '../../dataset/test'
        self.train_features = []
        self.test_features = []
        self.train_lables = []
        self.test_lables = []
        self.categories=categories

#Using image to vector extraction method that extends from the 'FeatureExtraction' Class
class Img2VecExtraction (_FeatureExtraction):

    def __init__(self,categories):
        self.__super__(categories)
        self.img2vec = Img2Vec()

    #Function for checking whether a image is grey scale
    def is_grey_scale_image(self,image):
        if image.mode == 'L':
            return True
        return False   
    
    #Convert grey scale image to color ones
    def to_three_channel_image(self,image):
        three_channel_image = image.convert('RGB')
        return three_channel_image

    #Reducing the amount of duplicata code and to loop through the image and store the features and lables of them
    def store_features_and_lables(self,lable,data_path,feature_array,lables_array):
        for image in os.listdir(data_path):
            train_image_path = os.path.join(data_path,image)
            train_img = Image.open(train_image_path)
            if self.is_grey_scale_image(train_img):
                train_img = self.to_three_channel_image(train_img)
            train_img_features = self.img2vec.get_vec(train_img)
            feature_array.append(train_img_features)
            lables_array.append(lable)

    
    def feature_extraction(self):
        
        for i in self.categories:

            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)
            
            self.store_features_and_lables(i,train_data_path,self.train_features,self.train_lables)

            self.store_features_and_lables(i,test_data_path,self.test_festures,self.test_lables)

class SIFTFeatureExtraction(_FeatureExtraction):
    def __init__(self,categories):
        self.super(categories)