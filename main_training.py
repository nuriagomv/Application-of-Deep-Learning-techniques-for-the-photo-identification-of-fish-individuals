# -*- coding: utf-8 -*-
"""
Created on June 2021

MAIN TRAINING

@author: Nuria Gómez-Vargas
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
os.chdir(r'C:\Users\CdeC\Desktop\AI iGENTAC - Nuria\algoritmo_final') # path to the folder of the .py files

import pickle

from siamese_network import SiameseNetwork


def main():

    dataset_path = r'C:\Users\CdeC\Desktop\AI iGENTAC - Nuria\datasets' #path where the folder raja_undulata is stored
    batch_size = 25

    load_dataset_again = True

    use_augmentation = True
    params_augmentation = {}
    if use_augmentation:
        params_augmentation['prob_aug'] = 0.50
        params_augmentation['by_channel'] = False
        params_augmentation['affine_scale'] = (1.05,1.2)
        params_augmentation['affine_trans'] = (0.001, 0.10)
        params_augmentation['sum'] = (1,50)
        params_augmentation['alpha_contrast'] = (1,2)
        params_augmentation['gauss_loc'] = (-30,30)
        params_augmentation['gauss_scale'] = (0,50)
        params_augmentation['times'] = 1
    
    learning_rate = 0.0351
    momentum = 0.6214
    std_prob_threshold = 0.05
    input_shape = (200, 250, 3)

    model_name = 'lr_' + str(learning_rate) + '__momentum_' + str(momentum) +  \
        '__std_prob_threshold_' + str(std_prob_threshold) + '__input_shape_' + str(input_shape)
            
    tensorboard_log_path = './logs/'+model_name

    siamese_network = SiameseNetwork(
        dataset_path = dataset_path,
        process = 'train',
        load_dataset_again = load_dataset_again,
        batch_size = batch_size,
        learning_rate = learning_rate,
        momentum = momentum,
        use_augmentation = use_augmentation,
        dict_augment = params_augmentation,
        input_shape = input_shape,
        tensorboard_log_path = tensorboard_log_path,
        std_prob_threshold = std_prob_threshold,
        model_name = model_name)

    print("\n Training model: ", model_name, "\n")

    number_of_train_iterations = 2000
    validate_each = 200

    validation_accuracy = siamese_network.train_siamese_network(number_of_iterations = number_of_train_iterations,
                                                                validate_each = validate_each,
                                                                model_name = model_name)
    
    # Once trained and validated, we make the predictions over the evaluation (test) set

    # Load the weights of the net, which are the ones with best validation accuracy
    siamese_network.model.load_weights('./models/'+model_name+'.h5')
        
    evaluation_accuracy, dict_save_predictions, dict_save_detections = siamese_network.predict_after_train()

    with open('PREDICCIONES__'+model_name+'.pkl','wb') as f:
         pickle.dump([evaluation_accuracy, dict_save_predictions, dict_save_detections], f)

    print('\n Final Evaluation Accuracy = ' + str(evaluation_accuracy))
    print('\n ======================================================================= \n \n ')

    siamese_network.matriz_confusion(model_name, dict_save_predictions, dict_save_detections)


if __name__ == "__main__":
    main()
