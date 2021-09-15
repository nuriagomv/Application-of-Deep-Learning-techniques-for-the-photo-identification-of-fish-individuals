# -*- coding: utf-8 -*-
"""
Created on June 2021

DATA AUGMENTATION

@author: Nuria Gómez-Vargas
"""

from imgaug import augmenters as iaa
from PIL import Image
import itertools
import numpy as np


class ImageAugmentor:
    """
    Class that performs image augmentation.

    Parameters:
    ----------
        prob_aug: 
            Probability of augmentation
        by_channel: bool
            Whether to sample per image only one value
        affine_scale: tuple
            Scaling factor to use.
        affine_trans: tuple
            Translation as a fraction of the image height/width.
        sum: tuple
            Add a value to all pixels in an image.
        alpha_contrast: tuple
            Multiplier to linearly pronounce, dampen or invert.
        gauss_loc: tuple
            Mean of the normal distribution from which the noise is sampled.
        gauss_scale: tuple
            Standard deviation of the normal distribution that generates the noise. 
        times: int
            Number of times to perform the augmentation.
        images_dictionary: dict
            dictionary with the images to augment
        image_width: int
            self explanatory
        image_height: int
            self explanatory
    """

    def __init__(self, images_dictionary, dict_params, image_height, image_width):
        """
        Inits ImageAugmentor with the provided values for the attributes.

        Parameters:
        ----------
            images_dictionary: dict
                dictionary with the images to augment
            dict_params: dict
                dictionary with the parameters for the data augmentation processes
            image_height: int
                self explanatory
            image_width: int
                self explanatory
        """

        self.prob_aug = dict_params['prob_aug']
        self.by_channel = dict_params['by_channel']
        self.affine_scale = dict_params['affine_scale']
        self.affine_trans = dict_params['affine_trans']
        self.sum = dict_params['sum']
        self.alpha_contrast = dict_params['alpha_contrast']
        self.gauss_loc = dict_params['gauss_loc']
        self.gauss_scale = dict_params['gauss_scale']
        self.times = dict_params['times']
        self.images_dictionary = images_dictionary
        self.image_width = image_width
        self.image_height = image_height


####################################################
# Transformations

    def padding(self, padding_percentage):
        """
        This function allows to preserve the complete image after the transformations. 

        Parameters
        ----------
        porcentaje: float
            If (0.5), double padding.
            If (-0.25), middle padding (back to original size).
            
        Returns
        -------
        imgaug.augmenters.size.CropAndPad : padding transformation.
        """
        pad = iaa.CropAndPad(percent = padding_percentage, sample_independently=False,
                             keep_size = False, pad_mode='constant', pad_cval=0)
        return pad


    def affine_transformation(self):
        """
        Transformaciones afines
        https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html#imgaug.augmenters.geometric.Affine
    
        Parameters
        ----------
        affine_scale: tuple
            Scaling factor to use. 
            scale = (a,b), and value will be uniformly sampled per image from the
            interval [a, b]. That value will be used identically for both x- and y-axis.

        affine_trans: tuple
            Translation as a fraction of the image height/width.
            trans = (a,b), and value will be uniformly sampled per image from the
            interval [a, b]. That value will be used identically for both x- and y-axis.
 
        Returns
        -------
        imgaug.augmenters.geometric.Affine : affine transformation.
    
        """
        ta = iaa.Affine(scale = self.affine_scale,
                        translate_percent = tuple([0.5*x for x in self.affine_trans]), # 0.5* because the size was doubled in the previous step
                        rotate = (-360, 360),
                        order = 1,
                        shear = (-45,45))
        return ta


    def sum_to_pixels(self):
        """
        Add a value to all pixels in an image.

        Parameters
        ----------
        sum: tuple
            sum = (a, b), then a value from the discrete interval [a..b] will be sampled per image.

        porCanal: bool
            Whether to use (imagewise) the same sample(s) for all channels (False) or to sample 
            value(s) for each channel (True). 
            
        Returns
        -------
        imgaug.augmenters.arithmetic.Add: transformation with the sum of the value to the pixels.
        """
        summ = iaa.Add(value = self.sum,
                      per_channel = self.by_channel)
        return summ


    def hue_sat_transformation(self):
        """
        Modifies saturation and hue by random values..
        The value is expected to be in the range -255 to +255.

        Parameters
        ----------
        by_channel: bool
            Whether to sample per image only one value from value and use it for both hue and 
            saturation (False) or to sample independently one value for hue and one for saturation (True). 
            
        Returns
        -------
        imgaug.augmenters.color.AddToHueAndSaturation: hue and saturation transformation.
        """
        ts = iaa.AddToHueAndSaturation(value_hue = (-255,255),
                                       value_saturation = (-255,255),
                                       per_channel=self.by_channel)
        return ts


    def contrast_transformation(self):
        """
        Adjust contrast by scaling each pixel to 127 + alpha*(v-127) where v is a pixel value.

        Parameters
        ----------
        alfa: tuple
            Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) 
            the difference between each pixel value and the dtype’s center value, e.g. 127 for uint8.

        porCanal: bool
            Whether to use the same value for all channels (False) or to sample a new value
            for each channel (True). 
            
        Returns
        -------
        imgaug.augmenters.contrast.LinearContrast: contrast transformation.
        """
        contr = iaa.LinearContrast(alpha=self.alpha_contrast,
                                   per_channel=self.by_channel)
        return contr


    def gaussNoise(self):
        """
        Add noise sampled from gaussian distributions elementwise to images.

        Parameters
        ----------
        loc: tuple
            Mean of the normal distribution from which the noise is sampled.

        scale: tuple
            Standard deviation of the normal distribution that generates the noise. 
            Must be >=0. If 0 then loc will simply be added to all pixels.

        by_channel: bool
            Whether to use the same value for all channels (False) or to sample a new value
            for each channel (True). 
            
        Returns
        -------
        imgaug.augmenters.arithmetic.AdditiveGaussianNoise: gaussian noise transformation.
        """
        gauss = iaa.AdditiveGaussianNoise(loc = self.gauss_loc, 
                                          scale = self.gauss_scale,
                                          per_channel = self.by_channel)
        return gauss


###################################################

    def randomAugmentation(self):
        """
        Applies transformations with probability prob_aug

        Parameters
        ----------
        prob_aug: float
            Probability with which the given augmenter will be applied.

        Returns
        -------
        imgaug.augmenters.size.KeepSizeByResize
        """

        random_aug = iaa.KeepSizeByResize(children = iaa.Sequential([
            self.padding(padding_percentage = (0.5)),
            iaa.Sometimes(self.prob_aug, self.sum_to_pixels()),
            iaa.Sometimes(self.prob_aug, self.hue_sat_transformation()),
            iaa.Sometimes(self.prob_aug, self.contrast_transformation()),
            iaa.Sometimes(self.prob_aug, self.gaussNoise()),
            iaa.Sometimes(self.prob_aug, self.affine_transformation()),
            self.padding(padding_percentage = (-0.25))]))

        return random_aug


    def apply_augmentation(self):
        """
        Applies augmentation to the images from the given dictionary.

        Parameters:
        ----------
        images_dictionary: dict
            dictionary with the images to augment

        Returns:
        -------
            dict: dictionary with the augmented images
        """

        print("Performing data augmentation " + str(self.times) + " times.\n")
        for ray in self.images_dictionary.keys():
            ray_dict = self.images_dictionary[ray]
            for (day,list_paths_photos) in ray_dict.items():
                list_photos = [np.array( Image.open(path).resize((self.image_width, self.image_height ), Image.ANTIALIAS)) for path in list_paths_photos]
                for _ in range(self.times):
                    images_aug = self.randomAugmentation().augment_images( list(list_photos) )
                    ray_dict[day] = list(itertools.chain.from_iterable( [ray_dict[day], images_aug] ))

                self.images_dictionary[ray] = ray_dict
    
        return self.images_dictionary
