import os
from PIL import Image
import matplotlib.pyplot as plt

def train_test_val(train_dim, val_dim, test_dim, tot):
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
                os.rename(os.path.join(original, path),
                          os.path.join(original + "train", path))
            elif (count > train_dim and count <= (train_dim+val_dim)):
                os.rename(os.path.join(original, path),
                          os.path.join(original + "validation", path))
            else:
                os.rename(os.path.join(original, path),
                          os.path.join(original + "test", path))
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
                os.rename(os.path.join(labels, path),
                          os.path.join(labels + "train", path))
            elif (count > train_dim and count <= (train_dim+val_dim)):
                os.rename(os.path.join(labels, path),
                          os.path.join(labels + "validation", path))
            else:
                os.rename(os.path.join(labels, path),
                          os.path.join(labels + "test", path))
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
                os.rename(os.path.join(original, path),
                          os.path.join(original_val, path))
            count += 1
            if count == 300:
                break

    print(count)

    count = 0
    for path in sorted(os.listdir(labels)):
        if os.path.isfile(os.path.join(labels, path)):
            if count > 200:
                os.rename(os.path.join(labels, path),
                          os.path.join(labels_val, path))
            count += 1
            if count == 300:
                break

    print(count)
"""


def data_augment(crop_size, dim_height, dim_width):
    # Data augmentation
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os
    import lib.cropping as cropping
    import lib.data_aug_photometric as aug
    import lib.data_aug_geometric as aug2

    original = "semantic_drone_dataset/original_images/train/"
    labels = "semantic_drone_dataset/label_images_semantic/train/"
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
                # do patch splitting
                # resize
                image = Image.open(os.path.join(r, i))
                # imgres = image.resize((3808,3808))
                img_list.append(image)

    # aug.data_aug_photometric(img_list,'.jpg',"semantic_drone_dataset/original_images/train")
    # aug2.data_aug_geometric(img_list,'.jpg',"semantic_drone_dataset/original_images/train")
    """
    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/train")):
        for i in f:
            if '.jpg' in i:
                # do patch splitting
                image = Image.open(os.path.join(r, i))
                img_cropped = cropping.crop_image(
                    image, crop_size, dim_width, dim_height)
                count = 0
                for k in img_cropped:
                    percorso = "semantic_drone_dataset/original_images/train/train_crop/"
                    percorso = percorso + str(os.path.splitext(i)[0])
                    percorso = percorso + "_"
                    percorso = percorso + str(count)
                    percorso = percorso + ".jpg"
                    k.save(percorso, 'JPEG')
                    count += 1
                os.remove(os.path.join(original, i))
    """
    seg_list = []
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/train")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                # segmres = segm.resize((3808, 3808))
                seg_list.append(segm)
    aug.seg_aug_photometric(seg_list,'.png','semantic_drone_dataset/label_images_semantic/train')
    aug2.data_aug_geometric(seg_list,'.png','semantic_drone_dataset/label_images_semantic/train')
    """
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/train")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                # segmres = segm.resize((3808, 3808))
                segm_cropped = cropping.crop_image(
                    segm, crop_size, dim_width, dim_height)
                count = 0
                for l in segm_cropped:
                    percorso2 = "semantic_drone_dataset/label_images_semantic/train/train_crop/"
                    percorso2 = percorso2 + str(os.path.splitext(j)[0])
                    percorso2 = percorso2 + "_"
                    percorso2 = percorso2 + str(count)
                    percorso2 = percorso2 + ".png"
                    l.save(percorso2, 'PNG')
                    count += 1
                os.remove(os.path.join(labels, j))

def data_augment_val(crop_size, dim_height, dim_width):
    # Data augmentation
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os
    import lib.cropping as cropping
    import lib.data_aug_photometric as aug
    import lib.data_aug_geometric as aug2

    original = "semantic_drone_dataset/original_images/validation/"
    labels = "semantic_drone_dataset/label_images_semantic/validation/"
    if not os.path.exists(original + "val_crop"):
        os.makedirs(original + "val_crop")
    if not os.path.exists(labels + "val_crop"):
        os.makedirs(labels + "val_crop")

    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/validation")):
        for i in f:
            if '.jpg' in i:
                # do patch splitting
                image = Image.open(os.path.join(r, i))
                img_cropped = cropping.crop_image(
                    image, crop_size, dim_width, dim_height)
                count = 0
                for k in img_cropped:
                    percorso = "semantic_drone_dataset/original_images/validation/val_crop/"
                    percorso = percorso + str(os.path.splitext(i)[0])
                    percorso = percorso + "_"
                    percorso = percorso + str(count)
                    percorso = percorso + ".jpg"
                    k.save(percorso, 'JPEG')
                    count += 1
                os.remove(os.path.join(original, i))
                
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/validation")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                # segmres = segm.resize((3808, 3808))
                segm_cropped = cropping.crop_image(segm, crop_size, dim_width, dim_height)
                count = 0
                for l in segm_cropped:
                    percorso2 = "semantic_drone_dataset/label_images_semantic/validation/val_crop/"
                    percorso2 = percorso2 + str(os.path.splitext(j)[0])
                    percorso2 = percorso2 + "_"
                    percorso2 = percorso2 + str(count)
                    percorso2 = percorso2 + ".png"
                    l.save(percorso2, 'PNG')
                    count += 1
                os.remove(os.path.join(labels, j))
                
def data_augment_test(crop_size, dim_height, dim_width):
    # Data augmentation
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os
    import lib.cropping as cropping
    import lib.data_aug_photometric as aug
    import lib.data_aug_geometric as aug2

    original = "semantic_drone_dataset/original_images/test/"
    labels = "semantic_drone_dataset/label_images_semantic/test/"
    
    if not os.path.exists(original + "test_crop"):
        os.makedirs(original + "test_crop")
    if not os.path.exists(labels + "test_crop"):
        os.makedirs(labels + "test_crop")

    for r, d, f in sorted(os.walk("semantic_drone_dataset/original_images/test")):
        for i in f:
            if '.jpg' in i:
                # do patch splitting
                image = Image.open(os.path.join(r, i))
                img_cropped = cropping.crop_image(
                    image, crop_size, dim_width, dim_height)
                count = 0
                for k in img_cropped:
                    percorso = "semantic_drone_dataset/original_images/test/test_crop/"
                    percorso = percorso + str(os.path.splitext(i)[0])
                    percorso = percorso + "_"
                    percorso = percorso + str(count)
                    percorso = percorso + ".jpg"
                    k.save(percorso, 'JPEG')
                    count += 1
                os.remove(os.path.join(original, i))
                
    for a, b, c in sorted(os.walk("semantic_drone_dataset/label_images_semantic/test")):
        for j in c:
            if '.png' in j:
                segm = Image.open(os.path.join(a, j))
                # segmres = segm.resize((3808, 3808))
                segm_cropped = cropping.crop_image(segm, crop_size, dim_width, dim_height)
                count = 0
                for l in segm_cropped:
                    percorso2 = "semantic_drone_dataset/label_images_semantic/test/test_crop/"
                    percorso2 = percorso2 + str(os.path.splitext(j)[0])
                    percorso2 = percorso2 + "_"
                    percorso2 = percorso2 + str(count)
                    percorso2 = percorso2 + ".png"
                    l.save(percorso2, 'PNG')
                    count += 1
                os.remove(os.path.join(labels, j))
    
def merge_image(img_list, crop_size, size_x, size_y):
    new_im = Image.new('RGB', (size_x, size_y))
    k = 0
    for i in range(0, int(size_y/crop_size)):
        for j in range(0, int(size_x/crop_size)):
            new_im.paste(img_list[k], (j*crop_size, i*crop_size))
            k = k+1
    return new_im

def unisci_immagine_jpg(directory, nameimg):
    patched_crop = []
    crop_names = []
    for path in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, path)):
            root_ext = os.path.splitext(path)
            stringa = root_ext[0].split("_")
            if stringa[0] == nameimg:
                crop_names.append(stringa[0])
                patched_crop.append(stringa[1])
    results = map(int, patched_crop)
    results = sorted(results)
    results = map(str, results)
    uniti = ",".join("{1}_{0}".format(x, y) for x, y in zip(results, crop_names))
    finale = []
    finale = uniti.split(",")
    finale2 = ",".join(directory+"/{0}.jpg".format(x) for x in finale)
    finale3 = []
    finale3 = finale2.split(",")
    lista_imm = []
    for i in finale3:
        lista_imm.append(Image.open(i))
    immagine = merge_image(lista_imm, 1000, 6000, 4000)
    return immagine


def unisci_immagine_png(directory, nameimg):
    patched_crop = []
    crop_names = []
    for path in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, path)):
            root_ext = os.path.splitext(path)
            stringa = root_ext[0].split("_")
            if stringa[0] == nameimg:
                crop_names.append(stringa[0])
                patched_crop.append(stringa[1])
    results = map(int, patched_crop)
    results = sorted(results)
    results = map(str, results)
    uniti = ",".join("{1}_{0}".format(x, y)
                     for x, y in zip(results, crop_names))
    finale = []
    finale = uniti.split(",")
    finale2 = ",".join(directory+"/{0}.png".format(x) for x in finale)
    finale3 = []
    finale3 = finale2.split(",")
    lista_imm = []
    for i in finale3:
        lista_imm.append(Image.open(i))
    immagine = merge_image(lista_imm, 1000, 6000, 4000)
    return immagine
