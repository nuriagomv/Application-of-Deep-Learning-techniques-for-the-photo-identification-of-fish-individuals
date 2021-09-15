# -*- coding: utf-8 -*-
"""
Created on June 2021

IMAGES LOADER

@author: Nuria Gómez-Vargas
"""

import os
import random
import numpy as np
import math
from PIL import Image
import pickle
import itertools

from tensorflow.keras.applications import inception_resnet_v2

from data_augment import ImageAugmentor


class Loader:
    """
    Class that loads and prepares the dataset of rays images.
    This Class was constructed to read the images and separate the training, validation and evaluation test.
    It also provides functions for geting one-shot task batches and performing the few-shots task.

    Parameters
    ----------
        dataset_path: str
            self explanatory
        batch_size: int
            number of rays chosen to pick a pair of similar and a pair of dissimilar photos
        image_height:int
            self explanatory
        image_width: int
            self explanatory
        use_augmentation: bool
            boolean that allows us to select if data augmentation is used or not
        dict_augment: dict
            dictionary with the parameters for the data augmentation processes
        std_prob_threshold: float
            threshold for the standard deviation of the probabilities of belonging to a category
        images_dictionary: dict
            dictionary of all the rays and their images.
        ray_names: list
            all the ray individuals.
        train_dictionary: dict
            dictionary of the images of the train set.
        source_to_evaluate: str
            name of the folder that is property of the IIM.
        validation_dictionary: dict
            dictionary of the images of the validation set (only rays from source_to_evaluate).
        evaluation_dictionary: dict
            dictionary of the images of the evaluation (test) set (only rays from source_to_evaluate).
        evaluable_rays: list
            names of the rays in the folder that is property of the IIM.
        n_eval: int
            number of images to retain for the evaluation set.
    """


    def __init__(self, dataset_path, process, load_dataset_again, batch_size, input_shape,
                 use_augmentation, dict_augment, std_prob_threshold, model_name):
        """
        Inits Loader with the provided values for the attributes.
        We load the images dataset and divide in train/validation/evaluation sets.

        Arguments:
            dataset_path: str
                self explanatory
            process: str
                distinction between train/prediction
            load_dataset_again: bool
                boolean that chooses to load data again or use previous train/valid/eval split
            batch_size: int
                number of rays chosen to pick a pair of similar and a pair of dissimilar photos
            input_shape: tuple
                image size
            use_augmentation: bool
                boolean that allows us to select if data augmentation is used or not
            dict_augment: dict
                dictionary with the parameters for the data augmentation processes
            std_prob_threshold: float
                threshold for the standard deviation of the probabilities of belonging to a category
        """

        self.dataset_path = dataset_path
        self.image_height = input_shape[0]
        self.image_width = input_shape[1]

        if 'train' in str(process):
            
            self.batch_size = batch_size
            self.use_augmentation = use_augmentation
            self.dict_augment = dict_augment
            self.std_prob_threshold = std_prob_threshold
            self.images_dictionary = {}
            self.ray_names = []
            self.train_dictionary = {}
            self.source_to_evaluate = 'raja_undulata'
            self.validation_dictionary = {}
            self.evaluation_dictionary = {}
            self.evaluable_rays = []

            if load_dataset_again:

                self.load_dataset()
                self.ray_names = list(self.images_dictionary.keys())

                if (self.use_augmentation):
                    
                    self.images_dictionary = ImageAugmentor(self.images_dictionary, self.dict_augment,
                                                            self.image_height, self.image_width).apply_augmentation()
        
                self.n_eval = 3 #para que tenga sentido el voto, que haya más de una foto
                self.divide_train_valid_eval()
                self.evaluable_rays = list(self.evaluation_dictionary.keys())

                # Not all the folders contain sufficient images. We retain only those rays that:
                # have two or more photos for training
                # if they are evaluable, have n_eval photos for evaluation and some image for validation
                for ray in list(self.ray_names.copy()):

                    if len(self.train_dictionary[ray]) <3:
                        print("Deleting " + ray)
                        del self.train_dictionary[ray]
                        try:
                            del self.validation_dictionary[ray]
                            del self.evaluation_dictionary[ray]
                        except:
                            None
                    
                    if ray in self.evaluable_rays:
                        try:
                            if (len(self.validation_dictionary[ray]) == 0) or (len(self.evaluation_dictionary[ray]) < self.n_eval):
                                print("Deleting " + ray)
                                del self.train_dictionary[ray]
                                del self.validation_dictionary[ray]
                                del self.evaluation_dictionary[ray]
                        except: #this will fail if the ray has been previously deleted
                            None

                with open('datasets'+model_name+'.pkl','wb') as f:
                    pickle.dump([self. images_dictionary, self.train_dictionary, self.validation_dictionary, self.evaluation_dictionary], f)

            else:
                with open('datasets'+model_name+'.pkl','rb') as f:
                    self.images_dictionary, self.train_dictionary, self.validation_dictionary, self.evaluation_dictionary = pickle.load(f)
        
            self.ray_names = list(self.train_dictionary.keys())
            self.evaluable_rays = list(self.evaluation_dictionary.keys())
            print("\n Finally, we have "+str(len(self.evaluable_rays))+" rays to photo-identify (i.e., categories)\n")

            total_train = 0
            for ray in self.ray_names:
                 print(ray + ': '+str(len( list(itertools.chain.from_iterable(self.images_dictionary[ray].values())) )) + ' images in total.')
                 if ray in os.listdir(os.path.join(self.dataset_path, self.source_to_evaluate)):
                     print("Of which " + str(len(self.evaluation_dictionary[ray])) + " are for evaluating (test set),")
                     print(str(len(self.train_dictionary[ray])) + " images are for the train set, and " + \
                         str(len(self.validation_dictionary[ray])) + " images are for the validation set.\n")
                 total_train += len(self.train_dictionary[ray])
            print("In total, my train set contains " + str(total_train) + " images of rays.\n")

        elif 'prediction' in str(process):

            with open('datasets'+model_name+'.pkl','rb') as f:
                self.images_dictionary, self.train_dictionary, self.validation_dictionary, self.evaluation_dictionary = pickle.load(f)
            self.ray_names = list(self.images_dictionary.keys())
            self.evaluable_rays = list(self.evaluation_dictionary.keys())


    def load_dataset(self):
        """
        Loads the dataset and stores the available images for each ray.
        """

        print("\nWe load the images data set:")
        total = 0
        for data_source in os.listdir(self.dataset_path):
            print("Extracting images from the source folder: " + data_source)
            source_path = os.path.join(self.dataset_path, data_source)
            for ray in os.listdir(source_path):
                if ('skate' in ray) and ('UNK' not in ray):
                    print("Extracting images of rays: " + ray)
                    ray_path = os.path.join(source_path, ray)
                    images = {}
                    for subfolder in os.listdir(ray_path):
                        subfolder_images = []
                        try:
                            subfolder_path = os.path.join(ray_path, subfolder)
                            for image in os.listdir(subfolder_path):
                                if image.lower().endswith('.jpg') or image.lower().endswith('.png'):
                                    image_path = os.path.join(subfolder_path, image)
                                    subfolder_images.append(image_path)
                        except: #for example for .xlsx files
                            None
                        images[subfolder] = subfolder_images
                        total += len(subfolder_images)
                    self.images_dictionary[ray] = images #AHORA IMAGES ES UN DICCIONARIO NO UNA LISTA. TENER EN CUENTA A LA HORA DE DIVIDE_TRAIN_VALID_EVAL
                    
        print("In total, we load "+str(total)+" images.\n")


    def divide_train_valid_eval(self):
        """
        The siamese architecture allows training with all the rays, which form the train set,
        and also choosing which of them are the categories we consider, which are the rays in the
        source_to_evaluate, and which will form the validation and evaluation sets.
        The function divides the images of each ray in the train, validation and evaluation (test) sets.
        First, it extracts n_eval images of each evaluable ray for evaluation.
        The rest of the images of the evaluable rays are divided in train and validation with a 75% - 25% split.
        """

        for ray in self.ray_names:

            ray_dict = self.images_dictionary[ray]

            if len(ray_dict) == 1: #if we dont have recaptures of this individual 

                train = list(ray_dict.values())[0].copy()
                if ray in os.listdir(os.path.join(self.dataset_path, self.source_to_evaluate)):  #we want to eval this individual later
                    self.evaluation_dictionary[ray] = []
                    if (len(train) > self.n_eval):
                        # If we sort the indexes in reverse order we can pop them from the list and the indexes won't change
                        inds_for_eval = random.sample(range(len(train)), self.n_eval)
                        inds_for_eval.sort(reverse=True)
                        eval = []
                        for ind in inds_for_eval:
                            eval.append( train.pop(ind) )
                        self.evaluation_dictionary[ray] = eval

                    inds_for_valid = random.sample(range(len(train)), int(0.25 * len(train)))
                    inds_for_valid.sort(reverse=True)
                    valid = []
                    for ind in inds_for_valid:
                        valid.append( train.pop(ind) )
                    self.validation_dictionary[ray] = valid

                self.train_dictionary[ray] = train
            
            else: # there are recaptures of this individual

                list_of_images = list(ray_dict.values())

                if ray in os.listdir(os.path.join(self.dataset_path, self.source_to_evaluate)): #we want to eval this individual later
                    lengths = [len(l) for l in list_of_images]
                    ind = lengths.index(min(lengths)) #test set is composed of the folder with the minimum number of photos
                    self.evaluation_dictionary[ray] = list_of_images.pop(ind)

                train = list(itertools.chain.from_iterable(list_of_images))

                if ray in os.listdir(os.path.join(self.dataset_path, self.source_to_evaluate)): #we want to eval this individual later
                    inds_for_valid = random.sample(range(len(train)), int(0.25 * len(train)))
                    inds_for_valid.sort(reverse=True)
                    valid = []
                    for ind in inds_for_valid:
                        valid.append( train.pop(ind) )
                    self.validation_dictionary[ray] = valid

                self.train_dictionary[ray] = train


    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        """
        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels.
        If the batch is from train or validation the labels are alternately 1's and 0's.
        If it is a validation/evaluation set only the first pair has label 1.

        Parameters
        ----------
            path_list: list
                list of images to be loaded in this batch
            is_one_shot_task: bool
                boolean signalizing if the batch is for one-shot task or if it is for training

        Returns:
        --------
            list: list of list pairs_of_images with pairs of images for the current batch
            list: correspondent labels, 1 for same class, 0 for different classes
        """
        
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            shape=(number_of_pairs, self.image_height, self.image_width, 3)) for _ in range(2)]
        labels = np.zeros(shape=(number_of_pairs, 1))

        for pair in range(number_of_pairs):
            try:
                for i in range(2):
                    instance = path_list[pair * 2 + i]
                    if isinstance(instance, str):
                        instance = instance.replace('CdeC\\Desktop','nuria\\OneDrive - UNIVERSIDAD DE SEVILLA\\Académico\\beca IIM-CSIC')
                        image = Image.open(instance)
                        image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
                    else: #it is not a path but a matrix (because it comes from the data augmentation)
                        image = instance

                    image = np.asarray(image).astype(np.float64)
                    #each Keras Application expects a specific kind of input preprocessing. 
                    image = inception_resnet_v2.preprocess_input(image)
                    pairs_of_images[i][pair, :, :, :] = image
            except Exception as e:
                print(str(e))
                print('Fail at loading some image of the pair...') #for example when the path has blanks

            if not is_one_shot_task:
                #in the list images_path, suppose [img1,img2,img3,img4,img5,img6,img7,img8],
                #if we see it as [par0,par1,par2,par3] (we count from 0, which is even)
                #we have that img1,img2,img5,img6 are of the same cateogry, which means label=1
                #this is, pairs with even index have label= 1
                #and img3,img4,img7,img8 are of different categories, which means label=0
                #this is, pairs with odd index have lavel= 0
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            else: #If it is a evaluation set (one_shot task) only the first pair has label 1
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task: #exclusive for training, we permute
            random_permutation = np.random.permutation(number_of_pairs)
            for i in range(2):
                pairs_of_images[i][:, :, :,:] = pairs_of_images[i][random_permutation, :, :, :]
            labels = labels[random_permutation]

        return pairs_of_images, labels


    def get_train(self):
        """
        Loads and returns the train images.
        It selects a number of ray individuals equals to batch_size. For each ray, we keep
        a pair of similar images (two images of the current ray) and a pair of dissimilar images 
        (consisting of an image of the current ray and an image of a different ray).

        Returns:
        --------
            list: list of list pairs_of_images with pairs of images for the current batch
            list: correspondent labels, 1 for same class, 0 for different classes
        """

        images_path = []

        selected_rays_indexes = [random.randint(0, len(self.ray_names)-1) for _ in range(self.batch_size)]

        for index in selected_rays_indexes:
            current_ray = self.ray_names[index]
            available_images_of_current_ray = self.train_dictionary[current_ray]
            
            # Random select 3 indexes of images from the same ray
            image_indexes = random.sample(range(len(available_images_of_current_ray)), 3)
            for ind in image_indexes:
                images_path.append( available_images_of_current_ray[ind] )
            
            # Now let's take care of the pair of images from different rays
            different_rays = self.ray_names.copy()
            different_rays.pop( self.ray_names.index(current_ray) )
            different_ray_index = random.randint(0, len(different_rays) - 1)
            current_different_ray = different_rays[different_ray_index]
            available_images_of_current_different_ray = self.train_dictionary[current_different_ray]
            image_index = random.randint(0, len(available_images_of_current_different_ray) - 1)
            image = available_images_of_current_different_ray[image_index]
            images_path.append(image)

        #finally, for each selected_ray, images_path contains for images,
        #the first 3 ones are of that ray, and the last one from a different ray
        #this is, a pair of same class images and a second pair of different class images

        images, labels = self._convert_path_list_to_images_and_labels(images_path, 
                                                                      is_one_shot_task = False)

        return images, labels


    def get_one_shot_batch(self, current_ray, test_image):
        """
        Loads and returns a batch for one-shot task images

        Gets a one-shot batch for validation set, it consists in a
        single test_image of the current_ray that will be compared with a 
        support set of images that we have previously trained on.
        It returns the pair of images to be compared by the model 
        and it's labels (the first pair is always 1) and the remaining ones are 0's.

        Parameters:
        -----------
            current_ray: str
                identification of the current_ray
            test_image: str
                path of the image of the ray to compare

        Returns:
        --------
            list: list of list pairs_of_images with pairs of images for the current batch
            list: correspondent labels, 1 for same class, 0 for different classes
        """

        #the comparison is against the support dictionary, but only with those evaluable rays
        support_dictionary = self.train_dictionary
        batch_images_path = []

        # Get another same ray image
        available_images = support_dictionary[current_ray]
        image_index = random.randint(0, len(available_images)-1)
        image = available_images[image_index]
        batch_images_path.append(test_image)
        batch_images_path.append(image)

        # Let's get our test image and its pair
        different_rays = list( set(self.evaluable_rays).difference(set([current_ray])) )
        for current_support_ray in different_rays:
            available_support_images = support_dictionary[current_support_ray]
            image_index = random.randint(0, len(available_support_images)-1)
            image = available_support_images[image_index]
            batch_images_path.append(test_image) #the test_image is added again before the suport image
            batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path, is_one_shot_task = True)

        return images, labels


    def few_shots_task(self, model):
        """
        Prepare few-shots task and evaluate its performance in validation and evaluation sets.

        Parameters:
        -----------
            model: keras.models.Model
            current siamese model

        Returns:
        --------
            mean_accuracy: mean accuracy for the validation few-shots task
        """

        print('\nMaking Few-Shots Task on validation images of the evaluable rays:')

        mean_rays_accuracies = []
        for ray in self.evaluable_rays:
            
            ray_accuracies = []
            for test_image in self.validation_dictionary[ray]:
                images, _ = self.get_one_shot_batch(current_ray = ray, test_image = test_image)
                probabilities = model.predict_on_batch(images)
                list_pred = [ind for (ind, value) in enumerate(probabilities) if value == max(probabilities)]
                if 0 in list_pred and probabilities.std() > self.std_prob_threshold:
                # Added this condition because noticed that sometimes the outputs
                # of the classifier was almost the same in all images.
                #If the probability is the maximum for the first (index=0) pair then the prediction 
                #is correct by definition of the list of pairs
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                ray_accuracies.append(accuracy)

            mean_ray_accuracy = np.mean(ray_accuracies)
            print(ray + ', validation accuracy: ' + str(mean_ray_accuracy))
            mean_rays_accuracies.append(mean_ray_accuracy)

        mean_global_accuracy = np.mean(mean_rays_accuracies)
        print('\nMean global validation accuracy: ' + str(mean_global_accuracy))

        return mean_global_accuracy
