import os

def train_test_val(train_dim,val_dim,test_dim,tot):
    original = "semantic_drone_dataset/original_images/"
    labels = "semantic_drone_dataset/label_images_semantic/"
    if not os.path.exists(original + "train"):
        os.makedirs(original + "train")
    if not os.path.exists(original + "validation"):
        os.makedirs(original + "validation")
    if not os.path.exists(original + "test"):
        os.makedirs(original + "test")

    count = 0
    for path in sorted(os.listdir(original)):
        if os.path.isfile(os.path.join(original, path)):
            if count <= train_dim:
                os.rename(os.path.join(original, path), os.path.join(original + "train", path))
            elif (count>train_dim and count<=(train_dim+val_dim)):
                os.rename(os.path.join(original, path), os.path.join(original + "validation", path))
            else:
                os.rename(os.path.join(original, path), os.path.join(original + "test", path))
            count += 1
            if count == tot:
                break

    print(count)

    if not os.path.exists(labels + "train"):
        os.makedirs(labels + "train")
    if not os.path.exists(labels + "validation"):
        os.makedirs(labels + "validation")
    if not os.path.exists(labels + "test"):
        os.makedirs(labels + "test")

    count = 0
    for path in sorted(os.listdir(labels)):
        if os.path.isfile(os.path.join(labels, path)):
            if count <= train_dim:
                os.rename(os.path.join(labels, path), os.path.join(labels + "train", path))
            elif (count>train_dim and count<=(train_dim+val_dim)):
                os.rename(os.path.join(labels, path), os.path.join(labels + "validation", path))
            else:
                os.rename(os.path.join(labels, path), os.path.join(labels + "test", path))
            count += 1
            if count == tot:
                break

    print(count)
"""
def validation_split():
    original = "semantic_drone_dataset/original_images/train/"
    original_val = "semantic_drone_dataset/original_images/val"
    labels = "semantic_drone_dataset/label_images_semantic/train/"
    labels_val = "semantic_drone_dataset/label_images_semantic/val"

    if not os.path.exists(original_val):
        os.makedirs(original_val)
    if not os.path.exists(labels_val):
        os.makedirs(labels_val)

    count = 0
    for path in sorted(os.listdir(original)):
        if os.path.isfile(os.path.join(original, path)):
            if count > 200:
                os.rename(os.path.join(original, path), os.path.join(original_val, path))
            count += 1
            if count == 300:
                break

    print(count)

    count = 0
    for path in sorted(os.listdir(labels)):
        if os.path.isfile(os.path.join(labels, path)):
            if count > 200:
                os.rename(os.path.join(labels, path), os.path.join(labels_val, path))
            count += 1
            if count == 300:
                break

    print(count)
"""
def data_augment():
    # Data augmentation
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os
    import lib.cropping as cropping
    import lib.data_aug_photometric as aug
    import lib.data_aug_geometric as aug2

    #original = "semantic_drone_dataset/original_images/train/"
    #labels = "semantic_drone_dataset/label_images_semantic/train/"
    if not os.path.exists(original + "train_crop"):
        os.makedirs(original + "train_crop")
    if not os.path.exists(labels + "train_crop"):
        os.makedirs(labels + "train_crop")

    """
    # r=root, d=directories, f = files
    img_list = []
    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/train")):
        for i in f:
            if '.jpg' in i:
                #do patch splitting
                #resize
                image = Image.open(os.path.join(r, i))
                #imgres = image.resize((3808,3808))
                img_list.append(image)
                
    #aug.data_aug_photometric(img_list,'.jpg',"semantic_drone_dataset/original_images/train")
    #aug2.data_aug_geometric(img_list,'.jpg',"semantic_drone_dataset/original_images/train")
    """
    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/train")):
        for i in f:
            if '.jpg' in i:
                #do patch splitting
                image = Image.open(os.path.join(r, i))
                img_cropped = cropping.crop_image(image,500,6000,4000)
                count = 0
                for k in img_cropped:
                    percorso = "semantic_drone_dataset/original_images/train/train_crop/"
                    percorso = percorso + str(os.path.splitext(i)[0])
                    percorso = percorso + "_"
                    percorso = percorso + str(count)
                    percorso = percorso + ".jpg"
                    k.save(percorso, 'JPEG')
                    count += 1
    """
    seg_list = []
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/train")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                #segmres = segm.resize((3808, 3808))
                seg_list.append(segm)
    aug.seg_aug_photometric(seg_list,'.png','semantic_drone_dataset/label_images_semantic/train')
    aug2.data_aug_geometric(seg_list,'.png','semantic_drone_dataset/label_images_semantic/train')
    """
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/train")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                #segmres = segm.resize((3808, 3808))
                segm_cropped = cropping.crop_image(segm, 500, 6000, 4000)
                count = 0
                for l in segm_cropped:
                    percorso2 = "semantic_drone_dataset/label_images_semantic/train/train_crop/"
                    percorso2 = percorso2 + str(os.path.splitext(j)[0])
                    percorso2 = percorso2 + "_"
                    percorso2 = percorso2 + str(count)
                    percorso2 = percorso2 + ".png"
                    l.save(percorso2, 'PNG')
                    count += 1

def data_augment_val():
    import os
    if not os.path.exists("semantic_drone_dataset/original_images/val/val_aug"):
        os.makedirs("semantic_drone_dataset/original_images/val/val_aug")
    if not os.path.exists("semantic_drone_dataset/label_images_semantic/val/val_aug"):
        os.makedirs("semantic_drone_dataset/label_images_semantic/val/val_aug")
    # Data augmentation
    from PIL import Image
    import cropping
    # r=root, d=directories, f = files
    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/val")):
        for i in f:
            if '.jpg' in i:
                # do patch splitting
                # resize
                image = Image.open(os.path.join(r, i))
                imgres = image.resize((3808, 3808))
                img_cropped = cropping.crop_image(imgres, 224, 3808, 3808)
                count = 0
                for k in img_cropped:
                    percorso = "semantic_drone_dataset/original_images/val/val_aug/"
                    percorso = percorso + str(os.path.splitext(i)[0])
                    percorso = percorso + "_"
                    percorso = percorso + str(count)
                    percorso = percorso + ".jpg"
                    k.save(percorso, 'JPEG')
                    count += 1

    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/val")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                segmres = segm.resize((3808, 3808))
                segm_cropped = cropping.crop_image(segmres, 224, 3808, 3808)
                count = 0
                for l in segm_cropped:
                    percorso2 = "semantic_drone_dataset/label_images_semantic/val/val_aug/"
                    percorso2 = percorso2 + str(os.path.splitext(j)[0])
                    percorso2 = percorso2 + "_"
                    percorso2 = percorso2 + str(count)
                    percorso2 = percorso2 + ".png"
                    l.save(percorso2, 'PNG')
                    count += 1
