from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as ks 
import numpy as np
from PIL import Image

def data_aug_geometric(list_img, extension_file, destination_path):
        
    k = 1
    
    for img in tqdm(list_img) : 
    
        #TRASFORMAZIONI DI FLIP ORIZZONTALE
        
        # convert to numpy array
        data = ks.img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(horizontal_flip=True)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1, seed = 2)
        # generate samples and plot
        for i in range(1):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(destination_path+"/file_"+str(k)+"_flipped_horiz_"+str(i)+extension_file, image)

        #TRASFORMAZIONI DI FLIP VERTICALE
        
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(vertical_flip=True)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1, seed = 2)
        # generate samples and plot
        for i in range(1):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(destination_path+"/file_"+str(k)+"_flipped_vert_"+str(i)+extension_file, image)


        #TRASFORMAZIONI DI ZOOM

        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1, seed = 2)
        # generate samples and plot
        for i in range(3):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(destination_path+"/file_"+str(k)+"_zoomed_"+str(i)+extension_file, image)
                
        k = k+1

