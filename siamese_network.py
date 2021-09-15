# -*- coding: utf-8 -*-
"""
Created on June 2021

SIAMESE NETWORK

@author: Nuria Gómez-Vargas
"""

import os
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD

from images_loader import Loader


class SiameseNetwork:
    """
    Class that constructs the siamese network and trains it.
    Also includes functions for prediction.

    Parameters
    ----------
        dataset_path: str
            self explanatory
        input_shape: tuple
            image size
        learning_rate: float
            SGD learning rate
        momentum: float
            SGD momentum
        process: str 
            distinction between train/prediction
        image_loader: images_loader.Loader
            instance of Loader
        summary_writer: tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
            tensorflow writer to store the logs
        model: keras.models.Model
            current siamese model
    """


    def __init__(self, dataset_path, process, load_dataset_again, batch_size,
                 learning_rate, momentum, use_augmentation,  dict_augment,
                 input_shape, tensorboard_log_path, std_prob_threshold, model_name):
        """
        Inits SiameseNetwork with the provided values for the attributes.
        It also constructs the siamese network architecture,
        creates a dataset loader and opens the log file.

        Parameters
        ----------
            dataset_path: str
                self explanatory
            process: str 
                distinction between train/prediction
            load_dataset_again: bool
                boolean that chooses to load data again or use previous train/valid/eval split
            batch_size: int
                number of rays chosen to pick a pair of similar and a pair of dissimilar photos
            learning_rate: float
                SGD learning rate
            momentum: float
                SGD momentum
            use_augmentation: bool
                boolean that allows us to select if data augmentation is used or not
            dict_augment: dict
                dictionary with the parameters for the data augmentation processes
            input_shape: tuple
                image size
            tensorboard_log_path: str
                path to store the logs
            std_prob_threshold: float
                threshold for the standard deviation of the probabilities of belonging to a category
        """
        
        print("Number of avaliable GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')), "\n")
        
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.process = process
        
        self.images_loader = Loader(
            dataset_path = self.dataset_path,
            process = self.process,
            load_dataset_again = load_dataset_again,
            batch_size = batch_size,
            input_shape = self.input_shape,
            use_augmentation = use_augmentation,
            dict_augment = dict_augment,
            std_prob_threshold = std_prob_threshold,
            model_name = model_name)

        if 'train' in str(self.process):
            self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)

        self._construct_siamese_architecture()


    def _construct_siamese_architecture(self):
        """
        Constructs the siamese architecture and stores it in the class.
        """

        # Let's define the cnn architecture for the verification task
        convolutional_net = Sequential(name = 'convolutional_twin_branch')
        # Load pretrained model
        pretrained_model = InceptionResNetV2(include_top = False, weights = 'imagenet',
                                             input_shape = self.input_shape, pooling = None)
        convolutional_net.add(pretrained_model)
        #freeze layer from pretrained model
        convolutional_net.get_layer("inception_resnet_v2").trainable = False
        # The units in the final convolutional layer are flattened into a single vector.
        convolutional_net.add(Flatten(name = 'vectorizing'))

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        #we compute the siamese twins
        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # elementwise L1 distance between the two encoded outputs
        l1_distance_layer = Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]),
                                   name = 'l1_distance_layer')
        # and then one more layer computing the induced distance metric between each siamese twin, 
        l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])
        # which is given to a single dense sigmoidal output unit.
        # Same class or not probability prediction
        prediction = Dense(units = 1, activation = 'sigmoid', name='output_prediction_layer')(l1_distance)

        self.model = Model(inputs = [input_image_1, input_image_2],
                           outputs = prediction, name='Siamese_model')

        if 'train' in str(self.process):
            print("\nSummary of the chosen pretrained model inception_resnet_v2:")
            pretrained_model.summary()
            print("\nSummary of the convolutional twin branch:")
            convolutional_net.summary()
            print("\nSummary of the siamese network:")
            print(self.model.summary())

            # Define the optimizer and compile the model
            optimizer = SGD(learning_rate = self.learning_rate, momentum = self.momentum, nesterov = False)
            self.model.compile(loss = 'binary_crossentropy', metrics = ['binary_accuracy'], optimizer = optimizer)


    def _write_logs_to_tensorboard(self, current_iteration, train_losses, train_accuracies,
                                   validation_accuracy, validate_each):
        """
        Writes the logs to a tensorflow log file.
        This allows us to see the loss curves and the metrics in tensorboard.
        We write the logs every validate_each iteration, not to slow the training process.

        Parameters
        ----------
            current_iteration: int
                iteration to be written in the log file
            train_losses: numpy.ndarray
                contains the train losses from the last validate_each iterations.
            train_accuracies: numpy.ndarray
                the same as train_losses but with the accuracies in the training set.
            validation_accuracy: float
                accuracy in the current few-shots task in the validation set
            validate_each: int
                number of iterations defined to validate the few-shots tasks.
        """

        # Write to log file the values from the last evaluate_every iterations
        for index in range(validate_each):

            epoch = current_iteration - validate_each + index + 1

            with self.summary_writer.as_default():
                tf.summary.scalar('Train Loss', train_losses[index], step=epoch)
                tf.summary.scalar('Train Accuracy', train_accuracies[index], step=epoch)
                if index == (validate_each - 1):
                    tf.summary.scalar('Few-shots Validation Accuracy', validation_accuracy, step=epoch)
            self.summary_writer.flush()


    def train_siamese_network(self, number_of_iterations, validate_each, model_name):
        """
        This is the main function for training the siamese net. 
        In each every validate_each train iterations we evaluate few-shots tasks in validation set.
        We also write to the log file.

        Parameters
        ----------
            number_of_iterations: int
                maximum number of iterations to train.
            validate_each: int
                number of iterations defined to validate the few-shots tasks.
            model_name: str
                save_name of the model

        Returns
        -------
            float: validation accuracy
        """

        # Variables that will store validate_each iterations losses and accuracies
        # after validate_each iterations these will be passed to tensorboard logs and reseted
        train_losses = np.zeros(shape = (validate_each))
        train_accuracies = np.zeros(shape = (validate_each))
        count = 0

        # Control variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0

        # train loop
        for iteration in range(number_of_iterations):

            # train set
            images, labels = self.images_loader.get_train()
            train_loss, train_accuracy = self.model.train_on_batch(images, labels)
            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, self.model.optimizer.lr))
            count += 1

            # Each validate_each iterations, perform a few_shots_task validation and
            # write to tensorboard the stored losses and accuracies
            if (iteration + 1) % validate_each == 0:
                validation_accuracy = self.images_loader.few_shots_task(model = self.model)
                
                self._write_logs_to_tensorboard(iteration, train_losses, train_accuracies,
                                                validation_accuracy, validate_each)
                count = 0

                # Save the model if we get a better validation accuracy
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration
                    
                    model_json = self.model.to_json()
                    
                    if not os.path.exists('./models'):
                        os.makedirs('./models')
                    with open('models/' + model_name + '.json', "w") as json_file:
                        json_file.write(model_json)
                    self.model.save_weights('models/' + model_name + '.h5')

        print('Train Ended!')
        return best_validation_accuracy


    def predict_after_train(self):
        """
        Function that evaluates the evaluation (test) set.
        The metodology resembles the ensamble methods techniques:
        - It predicts the category (identification) of every individual in the evaluation set
        by a majority vote of the predictions based on every image on its set.
        - These predictions of belonging to a category are actually the mean of the predictions
        obtained by comparing the test image with every image in the support (train) set of 
        each evaluable individual.

        Returns
        -------
            float: accuracy of the predictions
        """
        
        #List containing if the prediction was correct or not (True/False) for the latter evaluation accuracy
        correctness = []
        detections = []
        dict_save_predictions = {}
        dict_save_detections = {}

        #We predict every evaluable ray individual
        for current_ray in self.images_loader.evaluable_rays:

            print("\n Making predictions for ray: " + current_ray)
            predictions = []
            n_pred = 1
            current_images = self.images_loader.evaluation_dictionary[current_ray] #images to contrast against the support set
            inds_for_current = random.sample(range(len(current_images)), len(current_images)) #simply randomizing

            for i in inds_for_current:
                test_image = current_images[i]
                print("Prediction nº: "+str(n_pred))
                n_pred += 1
                
                support = {} #we reset the support set images for every prediction as we pop the images that have already been used
                for ray in self.images_loader.evaluable_rays: #because they are the only rays that we want to classify 
                    available_images = self.images_loader.train_dictionary[ray]
                    inds_for_support = random.sample(range(len(available_images)), len(available_images)) #simply randomizing
                    support[ray] = [available_images[j] for j in inds_for_support]

                list_of_predicted_probs = []
                continue_for_mean = True
                while continue_for_mean:

                    images_paths = []
                    for support_ray in support.keys():

                        images_paths.append(test_image)
                        try:
                            support_image = support[support_ray].pop()
                            images_paths.append( support_image )
                        except: # it will fail when, for some ray, the support[support_ray] list is empty 
                            continue_for_mean = False
                            images_paths = []
                            break

                    if images_paths != []:
                        images, _ = self.images_loader._convert_path_list_to_images_and_labels(images_paths, is_one_shot_task = True) #is_one_shot_task = True wont permutate the predictions
                        probabilities = self.model.predict_on_batch(images)
                        list_of_predicted_probs.append(probabilities)
                    
                final_mean_probs = [np.mean(list(probs)) for probs in zip(*list_of_predicted_probs)]
                max_prob = max(final_mean_probs)
                list_pred = [(ind,mean_prob) for (ind, mean_prob) in enumerate(final_mean_probs)
                             if mean_prob == max_prob]
                predictions += list_pred

            print("\n FINAL MAJORITY VOTE PREDICTION:")
            indxs_of_predictions = [p for (p,_) in predictions]
            times_voted = set([(ind, indxs_of_predictions.count(ind)) for ind in indxs_of_predictions])

            max_n_votes = max([times for (_, times) in times_voted])
            majority_vote_index = [ind for (ind, times) in times_voted if times == max_n_votes]
            majority_vote_prediction = [self.images_loader.evaluable_rays[ind] for ind in majority_vote_index]
            for indx_individual in majority_vote_index:
                prob_pred = np.mean([value for (ind,value) in predictions if ind == indx_individual])
                print("With mean probability = ", prob_pred, ", the majority vote prediction (voted "+ str(max_n_votes) +" times) is: " + self.images_loader.evaluable_rays[indx_individual])
            correct = current_ray in majority_vote_prediction
            print("Correct prediction? ", correct, "\n")
            correctness.append(correct)
            dict_save_predictions[current_ray] = majority_vote_prediction

            print('\n INDIVIDUAL AMONG TOP PREDICTIONS?')
            
            times_voted = [(ind,time) + (np.mean([value for (indx,value) in predictions if indx == ind]),) for (ind,time) in times_voted]
            majority_vote_order = sorted(times_voted, key=lambda x: (x[1],x[2]), reverse = True)
            majority_vote_detections = [(self.images_loader.evaluable_rays[ind], times, meanp) for (ind,times,meanp) in majority_vote_order if times >= math.ceil(len(self.images_loader.evaluation_dictionary[current_ray])/2) ]
            
            print('Top predictions:')
            detections_of_current_ray = {}
            for (detection, times, meanp) in majority_vote_detections:
                print(detection, 'times voted : '+str(times), 'with mean prob '+str(meanp))
                detections_of_current_ray[detection] = {'times': times, 'mean prob': meanp}
            detected = current_ray in [ray for (ray,_,_) in majority_vote_detections]
            print('\n The individual has been detected? ', detected)
            detections.append(detected)
            dict_save_detections[current_ray] = detections_of_current_ray

        mean_detections = np.mean(detections)

        print('\n PREDICTION OVER TEST SET ENDED! \n')

        print('\n Mean ACCURACY according to TOP predictions: ', mean_detections)

        accuracy = np.mean(correctness)

        return accuracy, dict_save_predictions, dict_save_detections


    def load_and_predict(self, images_to_predict):
        """
        Function that predicts over the dictionary.
        The metodology resembles the ensemble methods techniques:
        - It predicts the category (identification) of every individual in the evaluation set
        by a majority vote of the predictions based on every image on its set.
        - These predictions of belonging to a category are actually the mean of the predictions
        obtained by comparing the test image with every image in the support (train) set of 
        each evaluable individual.

        Parameters:
        ----------
        images_to_predict: dict
            dictionary with the images that we want to predict on.

        Returns
        -------
            numpy.ndarray: array flag_new_individuals with the paths of the individuals identified as new
        """
        
        #List that will contain the path of the individuals detected as new ones
        flag_new_individuals = []

        #We predict every evaluable ray individual
        for current_ray in images_to_predict.keys():

            print("\n Making predictions for ray: " + current_ray)
            predictions = []
            n_pred = 1
            current_images = images_to_predict[current_ray] #images to contrast against the support set
            inds_for_current = random.sample(range(len(current_images)), len(current_images)) #simply randomizing

            for i in inds_for_current:
                test_image = current_images[i]
                print("Prediction nº: "+str(n_pred))
                n_pred += 1
                
                support = {} #we reset the support set images for every prediction as we pop the images that have already been used
                for ray in self.images_loader.evaluable_rays: #because they are the only rays that we want to classify 
                    available_images = self.images_loader.train_dictionary[ray]
                    inds_for_support = random.sample(range(len(available_images)), len(available_images)) #simply randomizing
                    support[ray] = [available_images[j] for j in inds_for_support]

                list_of_predicted_probs = []
                continue_for_mean = True
                while continue_for_mean:

                    images_paths = []
                    for support_ray in support.keys():

                        images_paths.append(test_image)
                        try:
                            support_image = support[support_ray].pop()
                            images_paths.append( support_image )
                        except: # it will fail when, for some ray, the support[support_ray] list is empty 
                            continue_for_mean = False
                            images_paths = []
                            break

                    if images_paths != []:
                        images, _ = self.images_loader._convert_path_list_to_images_and_labels(images_paths, is_one_shot_task = True) #is_one_shot_task = True wont permutate the predictions
                        probabilities = self.model.predict_on_batch(images)
                        print('PRINTING PROBABILITIES GIVEN BY THE MODEL: ', probabilities)
                        list_of_predicted_probs.append(probabilities)
                    
                final_mean_probs = [np.mean(list(probs)) for probs in zip(*list_of_predicted_probs)]
                print('PRINTING FINAL MEAN PROBS: ', final_mean_probs)
                max_prob = max(final_mean_probs)
                list_pred = [(ind, mean_prob) for (ind, mean_prob) in enumerate(final_mean_probs) if mean_prob == max_prob]
                predictions += list_pred
                print("With mean probability = ",max_prob, ", the prediction in this comparison is: ",
                      [self.images_loader.evaluable_rays[pred] for pred in [p for (p,_) in list_pred]])
                
            print("\n FINAL MAJORITY VOTE PREDICTION \n")
            indxs_of_predictions = [p for (p,_) in predictions]
            times_voted = set([(ind, indxs_of_predictions.count(ind)) for ind in indxs_of_predictions])
            max_n_votes = max([times for (_, times) in times_voted])
            majority_vote_index = [ind for (ind, times) in times_voted if times == max_n_votes]
            majority_vote_prediction = [(self.images_loader.evaluable_rays[ind], max_n_votes, np.mean([value for (i,value) in predictions if i == ind])) for ind in majority_vote_index]
            for (individual, times, prob_pred) in majority_vote_prediction:
                if prob_pred < 1.0:
                    print("Flag: new individual identified")
                    majority_vote_prediction = None
                    flag_new_individuals.append(test_image) # It's sufficient with storing the last image because the path includes the individual code
                else:
                    print("With mean probability = ", prob_pred, ", the majority vote prediction (voted "+ str(times) +" times) is: " + individual)
            
            
            print('\n AND TOP PREDICTIONS ARE: \n')

            times_voted = [(ind,times) + (np.mean([value for (indx,value) in predictions if indx == ind]),) for (ind,times) in times_voted]
            majority_vote_order = sorted(times_voted, key=lambda x: (x[1],x[2]), reverse = True)
            if max_n_votes < math.ceil(len(images_to_predict[current_ray])/2):
                majority_vote_detections = majority_vote_prediction
            else:
                majority_vote_detections = [(self.images_loader.evaluable_rays[ind], times, meanp) for (ind,times,meanp) in majority_vote_order if times >= math.ceil(len(images_to_predict[current_ray])/2) ]
            
            print('Top predictions:')
            for (detection, times, meanp) in majority_vote_detections:
                print(detection, 'times voted : '+str(times), 'with mean prob '+str(meanp))

        return flag_new_individuals

   
    def process_new_individuals(self):
        """
        This function compare the new individuals in a pairwise manner
        to predict same/different individual for creating new categories.
        """

        store_path = os.path.join(self.dataset_path, 'folder_new_individuals')
        os.chdir(store_path)
        print(store_path)

        with open('file_new_individuals.txt', 'r') as f:
            paths = f.readlines()
            paths = [path[:-1] for path in paths if path!='\n']

        if len(paths) >= 2:
            print("\nThere are new individuals to categorize.\n")

            print("Creating pairs to compare in a pairwise manner...\n")
            pairs_of_images = [s for s in self.power_set(paths) if len(s)== 2]
            for pair in pairs_of_images:
                print("Predicting same/different over the pair:")
                print(pair)
                images, _ = self.images_loader._convert_path_list_to_images_and_labels(pair, is_one_shot_task = True)
                probability = self.model.predict_on_batch(images)[0][0]
                print('The probability of similarity is:', probability)
                
                if probability > 0.5:
                    print("The pair of images belong to the same individual")
                else:
                    print("The pair of images belong to different individuals")
                

    def power_set(self, c):
        """
        Computes the powert set of the set c
        """

        if len(c) == 0:
            return [[]]
        r = self.power_set(c[:-1])
        return r + [s + [c[-1]] for s in r]


    def matriz_confusion(self, model_name, dict_save_predictions, dict_save_detections):

        print('\n Confusion matrix for final predictions over test set \n')

        size = len(self.images_loader.evaluable_rays)

        #rows are real individual vs. columns are predicted individual
        matriz = pd.DataFrame(np.zeros((size,size)), index = self.images_loader.evaluable_rays, columns = self.images_loader.evaluable_rays, dtype = 'int64')

        for real_ray in dict_save_predictions.keys():
            for predicted_ray in dict_save_predictions[real_ray]:
                matriz.loc[real_ray,predicted_ray] += 1

        plt.figure()
        sns.heatmap(matriz, annot=False, cbar=True, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.title("Simplified Confusion Matrix"), plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        plt.savefig('confmatrix_'+model_name+'.png')
        #plt.show()

        print('\n Confusion matrix for top detections over test set \n')

        #rows are real individual vs. columns are predicted individual
        matriz_top = pd.DataFrame(np.zeros((size,size)), index = self.images_loader.evaluable_rays, columns = self.images_loader.evaluable_rays, dtype = 'int64')

        for real_ray in dict_save_detections.keys():
            detections_dict = dict_save_detections[real_ray]
            for detection in detections_dict.keys():
                times = detections_dict[detection]['times']
                n_images = len(self.images_loader.evaluation_dictionary[real_ray])
                matriz_top.loc[real_ray,detection] += times/n_images

        plt.figure()
        sns.heatmap(matriz_top, annot=False, cbar=True, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.title("Simplified Confusion Matrix for TOP detections"), plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        plt.savefig('matriz_confusion_top_detections'+model_name+'.png')
        #plt.show()
