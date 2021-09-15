# -*- coding: utf-8 -*-
"""
Created on June 2021

MAIN PREDICTION

@author: Nuria Gómez-Vargas
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
os.chdir(r'C:\Users\nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\beca IIM-CSIC\AI iGENTAC - Nuria\algoritmo_final')
from siamese_network import SiameseNetwork


def main():

    dataset_path = r'C:\Users\nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\beca IIM-CSIC\AI iGENTAC - Nuria\datasets'
    input_shape = (200, 250, 3)

    # We pick the model (i.e., the weights) with which we want to predict
    model_name = 'lr_0.096__momentum_0.845__std_prob_threshold_0.05__input_shape_(200, 250, 3)'

    # We set the architecture of our net
    siamese_network = SiameseNetwork(
        dataset_path = dataset_path,
        process = 'prediction',
        load_dataset_again = False,
        batch_size = 0,
        learning_rate = 0,
        momentum = 0,
        use_augmentation = False,
        dict_augment = {},
        input_shape = input_shape,
        tensorboard_log_path = '',
        std_prob_threshold = 0,
        model_name = model_name)
    
    print('PREDICTING UNKNOWN INDIVIDUALS with the weights of the model: '+model_name)
    siamese_network.model.load_weights('./models/'+model_name+'.h5')

    #loading unidentified individuals
    images_to_predict = {}
    source_to_evaluate = 'raja_undulata'
    source_path = os.path.join(dataset_path, source_to_evaluate)
    for ray in os.listdir(source_path):
        if ('UNK' in ray):
            print("Extracting images of rays: " + ray)
            ray_path = os.path.join(source_path, ray)
            images = []
            for subfolder in os.listdir(ray_path):
                try:
                    subfolder_path = os.path.join(ray_path, subfolder)
                    for image in os.listdir(subfolder_path):
                        if image.lower().endswith('.jpg') or image.lower().endswith('.png'):
                            image_path = os.path.join(subfolder_path, image)
                            images.append(image_path)
                except: #for example for .xlsx files
                    None
            images_to_predict[ray] = images

    flag_new_individuals = siamese_network.load_and_predict(images_to_predict)

    store_new_individuals(dataset_path, flag_new_individuals)

    siamese_network.process_new_individuals()
 

def store_new_individuals(dataset_path, paths):
    """
    This function stores in a .txt file the paths of the individuals identified as new.

    Parameters:
    ----------
    dataset_path: str
        self explanatory
    paths: list
        list with the predicted as new individuals paths of images
    """
    
    store_path = os.path.join(dataset_path, 'folder_new_individuals')
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    os.chdir(store_path)
    
    for path in paths:
        print("Saving image with path : ", path)
        with open('file_new_individuals.txt', "a") as f:
            f.write(path)
            f.write("\n")



if __name__ == "__main__":
    main()
