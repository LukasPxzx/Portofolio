import numpy as np
import cv2
import os

from skimage import feature

from .Feature_Extraction import _FeatureExtraction

class LBPExtraction(_FeatureExtraction):
    
    def __init__(self,categories):

        super().__init__(categories)
            

    def _store_features_and_labels(self,label,data_path,features,labels):

        radius = 1
        n_points = 8 * radius
            
        for image in os.listdir(data_path):

            image_path = os.path.join(data_path,image)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))

            lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= hist.sum()  #Normalize

            features.append(hist)
            labels.append(label)


    def feature_extraction(self):

        for i in self.categories:
            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)

            self._store_features_and_labels(i,train_data_path,self.train_features,self.train_lables)
            self._store_features_and_labels(i,test_data_path,self.test_features,self.test_lables)


        