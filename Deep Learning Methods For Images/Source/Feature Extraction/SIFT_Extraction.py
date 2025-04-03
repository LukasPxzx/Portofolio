import os
import cv2 
import numpy as np
from .Feature_Extraction import _FeatureExtraction

class SIFTExtraction(_FeatureExtraction):

    def __init__(self,categories):

        super().__init__(categories)

        #Create SIFT object
        self.sift = cv2.SIFT_create()

    def _flatten(self,features,labels):

        flatten_feature = []
        flatten_labels = []

        for i,desc in enumerate(features):
            if  desc.size > 0:
                for d in desc:
                    flatten_feature.append(d)
                    flatten_labels.append(labels[i])
                    
        return np.array(flatten_feature), np.array(flatten_labels)
    
    def _store_feature(self,data_path,curr_label,features,labels):

        temp_fea = []
        temp_lab = []

        for img in os.listdir(data_path):

            image_path = os.path.join(data_path,img)

            image = cv2.imread(image_path)

            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image,(64,64))

            _ , descriptors = self.sift.detectAndCompute(image,None)

            if descriptors is None:
                temp_fea.append(np.array([]))
            else:
                temp_fea.append(descriptors)

            temp_lab.append(curr_label)

        temp_fea,temp_lab = self._flatten(temp_fea,temp_lab)

        for i in temp_fea:
            features.append(i)
        
        for i in temp_lab:
            labels.append(i)
    

    def feature_extraction(self):
        for i in self.categories:

            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)

            self._store_feature(train_data_path,i,self.train_features,self.train_lables)
            self._store_feature(test_data_path,i,self.test_features,self.test_lables)