import numpy as np
import os
from img2vec_pytorch import Img2Vec
from PIL import Image
from .Feature_Extraction import _FeatureExtraction


class Img2VecExtraction (_FeatureExtraction):

    def __init__(self,categories):
        super().__init__(categories)
        self.img2vec = Img2Vec(model="resnet50")

    #Checking grey scale
    def _is_grey_scale_image(self,image):
        if image.mode == 'L':
            return True
        return False   
    
    #Convert grey scale image to color
    def _to_three_channel_image(self,image):
        three_channel_image = image.convert('RGB')
        return three_channel_image

    #Reducing the amount of duplicata code and to loop through the image and store the features and lables of them
    def _store_features_and_lables(self,lable,data_path,feature_array,lables_array):
        for image in os.listdir(data_path):
            train_image_path = os.path.join(data_path,image)
            train_img = Image.open(train_image_path)
            if self._is_grey_scale_image(train_img):
                train_img = self._to_three_channel_image(train_img)
            train_img_features = self.img2vec.get_vec(train_img)
            feature_array.append(train_img_features)
            lables_array.append(lable)
            
    #For converting str class labels to int format
    def class_label_str2int(self):

        int_train_labels = []
        int_test_labels = []
        
        
        for i in self.train_lables:
            for j in range(len(self.categories)):
                #print(self.categories[j])
                if self.categories[j] == i:
                    int_train_labels.append(j)
  
        
        for i in self.test_lables:

            for j in range(len(self.categories)):
                if self.categories[j] == i:
                    int_test_labels.append(j)
   

        return int_train_labels,int_test_labels

    def feature_extraction(self):
        
        for i in self.categories:

            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)
            
            self._store_features_and_lables(i,train_data_path,self.train_features,self.train_lables)

            self._store_features_and_lables(i,test_data_path,self.test_features,self.test_lables)