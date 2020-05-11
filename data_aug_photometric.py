from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from PIL import Image

def data_aug_photometric(list_img, extension_file, destination_path):

    k = 1
    
    for img in tqdm(list_img) :     

        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1, seed = 2)
        # generate samples and plot
        for i in range(3):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(destination_path+"/file_"+str(k)+"_brightnessed_"+str(i)+extension_file, image)

        k = k+1

def seg_aug_photometric(list_img, extension_file, destination_path):
    k=1
    for img in tqdm(list_img):
        for i in range(3):
            img.save(destination_path+"/file_"+str(k)+"_brightnessed_"+str(i)+extension_file,'PNG')
        k=k+1