
import tensorflow as tf
from keras_unet_collection import models
import os

data_path = '/home/vision/smb-datasets/med-datatsets/T4 Dataset'
# Configuración del generador de datos\n"
data_gen_kwargs = dict(rescale=1./255, validation_split=0.20)
image_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_kwargs)
mask_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_kwargs)
images_path = os.path.normpath(os.path.join(data_path, 'Images'))
masks_path = os.path.normpath(os.path.join(data_path, 'Masks'))
        
train_image_generator = image_data_gen.flow_from_directory(images_path, 
                                                            target_size=(128,128),
                                                            color_mode='grayscale',
                                                            class_mode=None,
                                                            batch_size=32,
                                                            shuffle=True,
                                                            seed=42,
                                                            subset='training')

train_mask_generator = mask_data_gen.flow_from_directory(masks_path,
                                                            target_size=(128,128),
                                                            color_mode='grayscale',
                                                            class_mode=None,
                                                            batch_size=32,
                                                            shuffle=True,
                                                            seed=42,
                                                            subset='training')

test_image_generator = image_data_gen.flow_from_directory(images_path,
                                                            target_size=(128,128),
                                                            color_mode='grayscale',
                                                            class_mode=None,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            seed=42,
                                                            subset='validation')

test_mask_generator = mask_data_gen.flow_from_directory(masks_path,
                                                            target_size=(128,128),
                                                            color_mode='grayscale',
                                                            class_mode=None,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            seed=42,
                                                            subset='validation')

# Crear generadores de datos combinados para entrenamiento y validación\n",
train_data_generator = zip(train_image_generator, train_mask_generator)
test_data_generator = zip(test_image_generator, test_mask_generator)

# for x in train_data_generator :
#     print("images - >  {}".format(x[0].shape))
#     print("--- masks - >  {}".format(x[1].shape))

import keras.backend as K

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true = tf.image.resize(y_true, (128, 128))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    up = (2. * intersection + smooth)
    down = (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return up / down
        
def bce_dice_loss(y_true, y_pred):
    celoss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return celoss + (1 - dice_coef(y_true, y_pred))


unet_2d = models.unet_2d(input_size=(128, 128, 1), filter_num=[64, 128, 256, 512], n_labels=1)
unet_2d.compile(optimizer=tf.keras.optimizers.Adam(), loss=bce_dice_loss, metrics=[dice_coef])
unet_2d.fit(train_data_generator, 
            epochs=10, 
            steps_per_epoch = 15,
            validation_data = test_data_generator)