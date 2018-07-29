#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:05:33 2018

@author: vmueller
"""
from keras.models import model_from_json

def save_model(model):
    # Serialize model to JSON
    try:
        regressor_json = model.to_json()
        
        with open("model.json", "w") as json_file:
            json_file.write(regressor_json)
        # Serialize weights to HDF5
        model.save_weights("model.h5")
        print("Model saved to disk")

    except:
        print("Not a Valid Model")
    
    


def load_model(file_name='model.json',weights_file='model.h5', optimizer='adam', loss='mean_squared_error'):
    # Load json and create model
    try:
        json_file = open(file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weigths into new model
        loaded_model.load_weights(weights_file)
        print("Loaded model from disk")
        loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        return loaded_model
    
    except:
        print("Not a Valid Model")
        
        return