from skimage.feature import hog
from PIL import Image
from .Feature_Extraction import _FeatureExtraction
import os
import cv2

class HOGExtraction(_FeatureExtraction):
    
    # Convert grey scale image to color ones
    def _to_three_channel_image(self,image):
        three_channel_image = image.convert('RGB')
        return three_channel_image
    
    def _store_features_and_labels(self,label,data_path,features,labels):
        
        for image in os.listdir(data_path):
            image_path = os.path.join(data_path,image)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, feature_vector=True)
            features.append(img_hog)
            labels.append(label)


    
    def feature_extraction(self):

        self._feature_extraction_super(self)

        # train_data_path = os.path.join(self.train_dir,i)
        # test_data_path = os.path.join(self.test_dir,i)

        # for i in self.categories:
        #     self._store_features_and_labels(i,train_data_path,self.train_features,self.train_lables)
        #     self._store_features_and_labels(i,test_data_path,self.test_lables,self.train_lables)
