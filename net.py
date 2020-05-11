import dataset_creator as d

d.train_test_50()
d.validation_split()
d.data_augment()
d.data_augment_val()

epochs = 5

from library_seg.library_seg.keras_segmentation.models.unet import resnet50_unet

n_classes = 23 # Aerial Semantic Segmentation Drone Dataset tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle
model2 = resnet50_unet(n_classes=n_classes,input_height=224,input_width=224)

from keras.optimizers import SGD
optim = SGD(learning_rate=0.0001,momentum=0.9,nesterov=True)

model2.train(
    train_images =  "semantic_drone_dataset/original_images/train/train_aug",
    train_annotations = "semantic_drone_dataset/label_images_semantic/train/train_aug",
    checkpoints_path = "unet" , epochs=epochs, ignore_zero_class=True,batch_size=16,
    optimizer_name=optim, val_images="semantic_drone_dataset/original_images/val/val_aug", val_annotations="semantic_drone_dataset/label_images_semantic/val/val_aug",validate=True,verify_dataset=False
)
#voglio salvare il modello creato
# serialize model to JSON
model_json = model2.to_json()
with open("mobilenet_unet_adam.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model2.save_weights("mobilenet_unet_adam.h5")
print("Saved model to disk")