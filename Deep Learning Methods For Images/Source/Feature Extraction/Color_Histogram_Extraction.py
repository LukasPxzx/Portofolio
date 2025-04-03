import cv2
import os

from .Feature_Extraction import _FeatureExtraction

class ColorHistogramExtraction(_FeatureExtraction):

    def __init__(self,categories):
        super().__init__(categories)
    

    def _compute_color_histogram(self,img_path, bins=(8, 8, 8)):
        #Load the image
        image = cv2.imread(img_path)
        #Convert to RGB (OpenCV loads images in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        
        #Compute the histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram and flatten it
        return hist
    
    def _store_features_and_labels(self,label,data_path,features,labels):
        
        for image in os.listdir(data_path):

            image_path = os.path.join(data_path,image)
            hist = self._compute_color_histogram(image_path)

            features.append(hist)
            labels.append(label)
    
    def feature_extraction(self):

        for i in self.categories:

            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)
            
            self._store_features_and_labels(i,train_data_path,self.train_features,self.train_lables)

            self._store_features_and_labels(i,test_data_path,self.test_features,self.test_lables)