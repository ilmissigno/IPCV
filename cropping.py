from tqdm import tqdm

def crop_image(img, crop_size, size_x, size_y):

    pbar = tqdm(total=((size_x/crop_size)*(size_y/crop_size)))
                
    x = 0
    y = 0
    z = crop_size
    v = crop_size
    finito = True
    cropped = []
    while finito == True:
        cropped.append(img.crop((x, y, z, v)))
        #cropped.append(img[x:x+z, y:y+v])
        img_cropped = cropped
        if x == size_x - crop_size:
            if y == size_y - crop_size and z == size_x and v == size_y:
                finito = False
            else:
                x = 0
                y = y + crop_size
                z = crop_size
                v = v + crop_size
        else:
            x = x + crop_size
            z = z + crop_size
    
        pbar.update(1)    
    
    pbar.close()
    
    return img_cropped
