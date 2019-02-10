#!/usr/bin/env python

##############
#### Eric Phan
##############

import numpy as np
import cgi
import re
import io
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/zacha/Downloads/ugahacks-food-identifier-3a0e6d73b5fc.json"
from sklearn import svm, metrics, neural_network
from skimage import io, feature, filters, exposure, color
from sklearn import preprocessing


from google.cloud import vision
from google.cloud.vision import types

client = vision.ImageAnnotatorClient()
form = cgi.FieldStorage()
uploadedImage = form.getValue('fileupload')

class ImageClassifier:
    
    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]


        
        return(data,labels)

    def extract_image_features(self, data):

        # extract feature vector from image data
        feature_data = []

        for image in data: 
            # content = image
            # newImage = vision.types.Image(content = content)
            # response = client.image_properties(newImage = newImage)
            # properties = response.image_properties_annotation
            # print('Properties of the image:')

            # for description in props.webDetection.webEntities:
            #     print('Fraction: {}'.format(webEntities.description))

            #image = color.rgb2gray(image)
            image = filters.gaussian(image, sigma = 1, multichannel = True)
            image = exposure.equalize_adapthist(image)
            #image = standard.fit_transform(image)
            image = feature.hog(image)
            
            feature_data.append(image)

            
            
        #pt.fit(feature_data)
        #feature_data = pt.transform(feature_data)
        
        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
 
        
        # train model and save the trained model to self.classifier
       
        #self.classifer = svm.LinearSVC()
        #self.classifer = self.classifer.fit(train_data, train_labels)
        self.classifer = neural_network.MLPClassifier(hidden_layer_sizes=150, activation= 'identity', early_stopping= True)
        self.classifer = self.classifer.fit(train_data, train_labels)
    

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        ########################
        # Please do not modify the return type below
        predicted_labels = self.classifer.predict(data)

        return predicted_labels

      
def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
