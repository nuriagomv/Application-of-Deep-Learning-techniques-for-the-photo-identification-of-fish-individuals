# -*- coding: utf-8 -*-
"""
Created on June 2021

BAYESIAN HYPERPARAMETER OPTIMIZATION

@author: Nuria Gómez-Vargas
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
os.chdir(r'C:\Users\CdeC\Desktop\AI iGENTAC - Nuria\algoritmo_final')

import GPyOpt
import tensorflow as tf

from siamese_network import SiameseNetwork


def main():

    def bayesian_optimization_function(x):

        dataset_path = r'C:\Users\CdeC\Desktop\AI iGENTAC - Nuria\datasets'
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
        
        current_learning_rate = float(x[:, 0])
        current_momentum = float(x[:, 1])
        current_std_prob_threshold = float(x[:, 2])
        current_input_shape = (int(x[:, 3]), int(x[:, 3]) + 50, 3)

        model_name = 'lr_' + str(current_learning_rate) + '__momentum_' + str(current_momentum) +  \
            '__std_prob_threshold_' + str(current_std_prob_threshold) + '__input_shape_' + str(current_input_shape)
        tensorboard_log_path = './logs/bho_nuevo_neval3/' + model_name
        
        siamese_network = SiameseNetwork(
            dataset_path = dataset_path,
            process = 'train',
            load_dataset_again = load_dataset_again,
            batch_size = batch_size,
            learning_rate = current_learning_rate,
            momentum = current_momentum,
            use_augmentation = use_augmentation,
            dict_augment = params_augmentation,
            input_shape = current_input_shape,
            tensorboard_log_path = tensorboard_log_path,
            std_prob_threshold = current_std_prob_threshold,
            model_name = model_name)

        print("Training model: ", model_name, "\n")

        number_of_train_iterations = 1500
        validate_each = 250
        
        validation_accuracy = siamese_network.train_siamese_network(number_of_iterations = number_of_train_iterations,
                                                                    validate_each = validate_each,
                                                                    model_name = model_name)

        # Once trained and validated, we make the predictions over the evaluation (test) set

        # Load the weights of the net, which are the ones with best validation accuracy
        siamese_network.model.load_weights('./models/'+model_name+'.h5')
        
        evaluation_accuracy, _, _ = siamese_network.predict_after_train()
        print("Model: " + model_name + ' | Accuracy: ' + str(evaluation_accuracy), "\n")

        return 1 - evaluation_accuracy


    hyperparameters = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (10e-5, 10e-2)},
                       {'name': 'momentum', 'type': 'continuous', 'domain': (0.0, 1.0)},
                       {'name': 'std_prob_threshold', 'type': 'discrete', 'domain': (0.05, 0.25)},
                       {'name': 'input_shape', 'type': 'discrete', 'domain': (75,100,200)}]

    optimizer = GPyOpt.methods.BayesianOptimization(f = bayesian_optimization_function,
                                                    domain = hyperparameters)

    max_iter_for_run_optimization = 100

    optimizer.run_optimization(max_iter = max_iter_for_run_optimization)

    print("optimized parameters: {0}".format(optimizer.x_opt))
    print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))


if __name__ == "__main__":
    main()
