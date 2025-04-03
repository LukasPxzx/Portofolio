import os

class _FeatureExtraction:
    
    def __init__(self,categories):
        self.train_dir = '../../dataset/train' 
        self.test_dir = '../../dataset/test'
        self.train_features = []
        self.test_features = []
        self.train_lables = []
        self.test_lables = []
        self.categories=categories
    
    
    def _feature_extraction_super(self, child_obj):

        for i in self.categories:

            train_data_path = os.path.join(self.train_dir,i)
            test_data_path = os.path.join(self.test_dir,i)
            
            child_obj._store_features_and_labels(i,train_data_path,self.train_features,self.train_lables)

            child_obj._store_features_and_labels(i,test_data_path,self.test_features,self.test_lables)
