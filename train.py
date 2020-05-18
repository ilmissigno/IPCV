from lib_1.keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
from keras.callbacks import Callback

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          img_path_train,
          img_path_val,
          seg_path_train,
          seg_path_val,
          n_classes,
          validate=True,
          checkpoints_path=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          epochs=5,
          gen_use_multiprocessing=True,
          optimizer_name='adadelta'):

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

    input_shape = model.input_shape
    output_shape = model.output_shape
    train_gen = image_segmentation_generator(images_path=img_path_train,
                                             segs_path=seg_path_train,
                                             batch_size=2, n_classes=n_classes, input_height=input_shape[1],
                                             input_width=input_shape[2],
                                             output_height=output_shape[1], output_width=output_shape[2],
                                             do_augment=False, augmentation_name=None)
    val_gen = image_segmentation_generator(images_path=img_path_val,
                                           segs_path=seg_path_val,
                                           batch_size=2,
                                           n_classes=n_classes,
                                           input_height=input_shape[1],
                                           input_width=input_shape[2],
                                           output_height=output_shape[1],
                                           output_width=output_shape[2])

    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]

    if not validate:
        model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing)