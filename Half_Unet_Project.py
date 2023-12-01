import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from keras.layers import *
from keras import backend as K
import os
import random
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
DRIVE_PATH = '/content/gdrive'
FOLDER_PATH = r'D:\game\Processed_Dataset'
drive.mount(DRIVE_PATH)


# Creating custom data generator for data loading
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=2, image_size=256):
        self.ids = ids
        self.path = FOLDER_PATH + path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "images", id_name)

        ## Reading Image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0
        return image

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]
        image = []
        for id_name in files_batch:
            # _img, _mask = self.__load__(id_name)
            _img = self.__load__(id_name)
            image.append(_img))
            image = np.array(image)
        return image

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def check_images():
    image_size = 256
    train_path = FOLDER_PATH + "training/images"
    test_path = FOLDER_PATH + "testing/images"
    validation_path = FOLDER_PATH + "validation/images"
    epochs = 10
    batch_size = 2

    ## Training Ids

    train_ids = os.listdir(train_path)
    validation_ids = os.listdir(validation_path)
    test_ids = os.listdir(test_path)
    gen = DataGen(train_ids, "training/", batch_size=batch_size, image_size=image_size)
    x, y = gen.__getitem__(0)
    print(x.shape, y.shape)
    r = random.randint(0, len(x) - 1)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.reshape(x[0], (image_size, image_size, 3)), cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[0], (image_size, image_size)), cmap="gray")
    plt.show()

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.reshape(x[1], (image_size, image_size, 3)), cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[1], (image_size, image_size)), cmap="gray")
    plt.show()


print(os.listdir(FOLDER_PATH))
check_images()

def train_data_aug(batch_size = 2):
    seed = 1
    image_datagen = ImageDataGenerator(rotation_range=0.2, rescale=1./255, width_shift_range=0.05,
                    height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                    horizontal_flip=True, fill_mode='nearest')
    #mask_datagen = ImageDataGenerator(rotation_range=0.2, rescale=1./255, width_shift_range=0.05,
    #                height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
    #                horizontal_flip=True, fill_mode='nearest')
    dir = FOLDER_PATH + 'training/'
    image_generator = image_datagen.flow_from_directory(dir + 'images', class_mode=None, seed=seed,
                    color_mode="grayscale", target_size=(256,256), batch_size=batch_size)
    #mask_generator = mask_datagen.flow_from_directory(dir + 'mask', class_mode=None, seed=seed,
    #               color_mode="grayscale", target_size=(256,256), batch_size=batch_size)
    #train_generator = zip(image_generator, mask_generator)
    train_generator = zip(image_generator)
    return train_generator

def test_data_aug():
    seed = 1
    image_datagen = ImageDataGenerator(rescale=1./255)
    #mask_datagen = ImageDataGenerator(rescale=1./255)
    dir = FOLDER_PATH + 'testing/'
    image_generator = image_datagen.flow_from_directory(dir + 'images', shuffle=False, class_mode=None,
                    seed=seed, color_mode="grayscale", target_size=(256,256), batch_size=1)
    #mask_generator = mask_datagen.flow_from_directory(dir + 'mask', shuffle=False, class_mode=None,
    #                seed=seed, color_mode="grayscale", target_size=(256,256), batch_size=1)
    #test_generator = zip(image_generator, mask_generator)
    test_generator = zip(image_generator)
    return test_generator

def check_data_generator_images(c = 0):
    f, axarr = plt.subplots(1,2, figsize=(15, 15))
    for i in train_data_aug():
        if c >= 20:
            break
    im = i[0][0][:,:,0]
    ms = i[1][0][:,:,0]
    axarr[0].imshow(im)
    axarr[1].imshow(ms)
    c += 1
plt.show()


# code for Half-UNet architecture based on the research paper
def ghost_module(inputs):
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    conv2 = SeparableConv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(act1)
    return concatenate([act1, conv2], axis=3)


def model(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape, name="image")
    x1 = ghost_module(ghost_module(inputs))
    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = ghost_module(ghost_module(pool1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = ghost_module(ghost_module(pool2))
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = ghost_module(ghost_module(pool3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = ghost_module(ghost_module(pool4))

    up5 = UpSampling2D(size=(input_shape[0] // x5.shape[1], input_shape[1] // x5.shape[2]))(x5)
    up4 = UpSampling2D(size=(input_shape[0] // x4.shape[1], input_shape[1] // x4.shape[2]))(x4)
    up3 = UpSampling2D(size=(input_shape[0] // x3.shape[1], input_shape[1] // x3.shape[2]))(x3)
    up2 = UpSampling2D(size=(input_shape[0] // x2.shape[1], input_shape[1] // x2.shape[2]))(x2)

    upScaled = Add()([x1, up2, up3, up4, up5])
    all_conv = ghost_module(ghost_module(upScaled))
    final_conv = Conv2D(1, 1, activation='sigmoid')(all_conv)

    # final_conv = Conv2D(2, (1, 1), activation = 'softmax')(all_conv)
    half_unet_model = tf.keras.Model(inputs, final_conv, name="Half-UNet")
    return half_unet_model

#code for base version of UNet Architecture
from keras.models import Model

def main_conv_block(input, num):
    conv = Activation("relu")(BatchNormalization()(Conv2D(num, 3, padding="same")(input)))
    conv = Activation("relu")(BatchNormalization()(Conv2D(num, 3, padding="same")(conv)))
    return conv

def left_block(input, num):
    conv = main_conv_block(input, num)
    pool = MaxPool2D((2, 2))(conv)
    return conv, pool

def right_block(input, skip_connect, num):
    conv = main_conv_block(Concatenate()([Conv2DTranspose(num, (2, 2), strides=2, padding="same")(input), skip_connect]), num)
    return conv

def unet_model(input_shape = (256, 256, 3)):
    inputs = Input(input_shape)

    conv1, pool1 = left_block(inputs, 64)
    conv2, pool2 = left_block(pool1, 128)
    conv3, pool3 = left_block(pool2, 256)
    conv4, pool4 = left_block(pool3, 512)

    bottlneck = main_conv_block(pool4, 1024)

    out1 = right_block(bottlneck, conv4, 512)
    out2 = right_block(out1, conv3, 256)
    out3 = right_block(out2, conv2, 128)
    out4 = right_block(out3, conv1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(out4)

    model = Model(inputs, outputs, name="U-Net")
    return model


# code for Nested Half-UNet architecture based on U^2-Net

def half_unet_block(inputs):
    x1 = ghost_module(ghost_module(inputs))
    return x1, MaxPooling2D(pool_size=(2, 2))(x1)


def half_unet_model_modified(inp, depth=5, input_shape=256):
    all_skips = []
    for dp in range(depth):
        x, p = half_unet_block(inp)
        inp = p
        all_skips.append(x)
    layers_to_add = []
    for i, skip in enumerate(all_skips):
        if i == 0:
            layers_to_add.append(skip)
            continue
    up = UpSampling2D(size=(input_shape // skip.shape[1], input_shape // skip.shape[2]))(skip)
    layers_to_add.append(up)

    upScaled = Add()(layers_to_add)
    return upScaled
    # return ghost_module(ghost_module(upScaled))


def nested_half_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape, name="image")
    x1 = half_unet_model_modified(inputs, depth=4)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = half_unet_model_modified(pool1, depth=3, input_shape=128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = half_unet_model_modified(pool2, depth=3, input_shape=64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = half_unet_model_modified(pool3, depth=2, input_shape=32)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = half_unet_model_modified(pool4, depth=1, input_shape=16)

    up5 = UpSampling2D(size=(input_shape[0] // x5.shape[1], input_shape[1] // x5.shape[2]))(x5)
    up4 = UpSampling2D(size=(input_shape[0] // x4.shape[1], input_shape[1] // x4.shape[2]))(x4)
    up3 = UpSampling2D(size=(input_shape[0] // x3.shape[1], input_shape[1] // x3.shape[2]))(x3)
    up2 = UpSampling2D(size=(input_shape[0] // x2.shape[1], input_shape[1] // x2.shape[2]))(x2)
    upScaled = Add()([x1, up2, up3, up4, up5])
    all_conv = half_unet_model_modified(upScaled)

    final_conv = Conv2D(1, 1, activation='sigmoid')(all_conv)

    # final_conv = Conv2D(2, (1, 1), activation = 'softmax')(all_conv)
    half_unet_model = tf.keras.Model(inputs, final_conv, name="Half-UNet")
    return half_unet_model

half_unet_model = model((256, 256, 3))
half_unet_model.summary()

unet_model = unet_model((256, 256, 3))
unet_model.summary()

nested_half_unet = nested_half_unet((256, 256, 3))
nested_half_unet.summary()

#code for visualizing all the metrics and the loss during training
import matplotlib.pyplot as plt, random, numpy as np, cv2
from PIL import Image
from keras import backend as K

def training_history_plot(results):
	titles = ["dice_loss",'accuracy', "iou", "F1", "recall", "precision", "dice_coef"]
	metric = ['loss', 'accuracy', 'iou','F1','recall','precision','dice_coef'] # Metrics we're keeping track off
	val_metric = ['val_loss', 'val_accuracy', 'val_iou','val_F1','val_recall','val_precision','val_dice_coef']
	# Define specification of our plot
	fig, axs = plt.subplots(4,2, figsize=(15, 15), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.5, wspace=0.2)
	axs = axs.ravel()

	for i in range(7):
		axs[i].plot(results.history[metric[i]]) # Calls from 'History.history'- 'metric[i]', note 'results' is
		axs[i].plot(results.history[val_metric[i]])
		axs[i].set_title(titles[i])				# a 'History' object
		axs[i].set_xlabel('epoch')
		axs[i].set_ylabel(metric[i])
		axs[i].legend(['train'], loc='upper left')

#custom loss and metrics functions
def iou(y_true, y_pred, smooth=1):
	intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
	union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
	iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
	return iou

def F1(y_true, y_pred, smooth=1):
	intersection = K.sum(y_true * y_pred, axis=[1,2,3])
	union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
	dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
	return dice

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def dice_coef(y_true, y_pred, smooth = 0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def scheduler(epoch, lr):
      if epoch < 30:
        lr = 0.001
        return lr
      if epoch < 50:
        return 0.0005
      return 0.0001

#initializing data generators
btch_size = 2
image_size = 256

train_path = FOLDER_PATH + "training/images"
test_path = FOLDER_PATH + "testing/images"
validation_path = FOLDER_PATH + "validation/images"

train_ids = os.listdir(train_path)
validation_ids = os.listdir(validation_path)
test_ids = os.listdir(test_path)

#training code for half-UNet (The output is for the base half-UNet model)
train_gen = DataGen(train_ids, 'training/', image_size=image_size, batch_size=btch_size)
valid_gen = DataGen(validation_ids, 'validation/', image_size=image_size, batch_size=btch_size)
train_steps = len(train_ids)//btch_size
valid_steps = len(validation_ids)//btch_size
half_unet_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', iou, F1, recall, precision, dice_coef])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = half_unet_model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=60, callbacks=[callback])
training_history_plot(history)
half_unet_model.save('half_unet')

#training code for UNet
train_gen = DataGen(train_ids, 'training/', image_size=image_size, batch_size=btch_size)
valid_gen = DataGen(validation_ids, 'validation/', image_size=image_size, batch_size=btch_size)
train_steps = len(train_ids)//btch_size
valid_steps = len(validation_ids)//btch_size
unet_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', iou, F1, recall, precision, dice_coef])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = unet_model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=60, callbacks=[callback])
training_history_plot(history)
unet_model.save('unet')

#training code for Nested half-UNet
train_gen = DataGen(train_ids, 'training/', image_size=image_size, batch_size=btch_size)
valid_gen = DataGen(validation_ids, 'validation/', image_size=image_size, batch_size=btch_size)
train_steps = len(train_ids)//btch_size
valid_steps = len(validation_ids)//btch_size
nested_half_unet.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', iou, F1, recall, precision, dice_coef])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = nested_half_unet.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=60, callbacks=[callback])
training_history_plot(history)
nested_half_unet.save('nested_half_unet')

#training code for half-UNet (The output is for the half-UNet model with L2 Regularization)
train_gen = DataGen(train_ids, 'training/', image_size=image_size, batch_size=btch_size)
valid_gen = DataGen(validation_ids, 'validation/', image_size=image_size, batch_size=btch_size)
train_steps = len(train_ids)//btch_size
valid_steps = len(validation_ids)//btch_size
half_unet_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', iou, F1, recall, precision, dice_coef])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = half_unet_model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=60, callbacks=[callback])
training_history_plot(history)
half_unet_model.save('half_unet_reg')

#training code for half-UNet (The output is for the half-UNet model with Batch Normalization)
train_gen = DataGen(train_ids, 'training/', image_size=image_size, batch_size=btch_size)
valid_gen = DataGen(validation_ids, 'validation/', image_size=image_size, batch_size=btch_size)
train_steps = len(train_ids)//btch_size
valid_steps = len(validation_ids)//btch_size
half_unet_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', iou, F1, recall, precision, dice_coef])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = half_unet_model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=60, callbacks=[callback])
training_history_plot(history)
half_unet_model.save('half_unet_norm')


all_mods = ['half_unet', 'half_unet_reg', 'half_unet_norm', 'unet', 'nested_half_unet']

btch_size = 2
image_size = 256

train_path = FOLDER_PATH + "training/images"
test_path = FOLDER_PATH + "testing/images"
validation_path = FOLDER_PATH + "validation/images"

train_ids = os.listdir(train_path)
validation_ids = os.listdir(validation_path)
test_ids = os.listdir(test_path)

def iou(y_true, y_pred, smooth=1):
	intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
	union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
	iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
	return iou

def F1(y_true, y_pred, smooth=1):
	intersection = K.sum(y_true * y_pred, axis=[1,2,3])
	union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
	dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
	return dice

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def dice_coef(y_true, y_pred, smooth = 0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def load_model(model_name):
  #loading model
  return keras.models.load_model(FOLDER_PATH + "Trained_Models/" + model_name, custom_objects ={'iou':iou, 'F1':F1, 'recall':recall, 'precision':precision, 'dice_coef':dice_coef})

def load_models(model_name = 'all'):
  #loading all or specific model based on choice
  if model_name == 'all':
    nested_half_unet = load_model('nested_half_unet')
    half_unet_reg = load_model('half_unet_reg')
    half_unet = load_model('half_unet')
    half_unet_norm = load_model('half_unet_norm')
    unet = load_model('unet')
    print("ALL Models Loaded Successfully")
    return [half_unet, half_unet_reg, half_unet_norm, unet, nested_half_unet]
  print(model_name, "Loaded Successfully")
  return load_model(model_name)

def load_test_data(batch_size = 10):
  #loading test generator
  test_ids = os.listdir(test_path)
  gen = DataGen(test_ids, "testing/", batch_size=batch_size, image_size=image_size)
  print("Test Data Loaded")
  return gen, len(test_ids)

def visualize_output(x, y, y_pred, tot, idx, fig, title_flag = False):
  #visualizing model outputs from different models
  contours1, hierarchy = cv2.findContours(y[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours2, hierarchy = cv2.findContours(y_pred[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(x[0], contours1, -1, (0, 255, 0), 3)
  cv2.drawContours(x[0], contours2, -1, (255, 0, 0), 3)
  ax = fig.add_subplot(tot, 5, idx)
  # if title_flag:
  ax.set_title(all_mods[idx - 1], fontdict={'fontsize': 20, 'fontweight': 'medium'})
  ax.imshow(np.reshape(x[0], (image_size, image_size, 3)))

def visualize_outputs_of__all_models(gen, all_models, start = 0, test_img_count = 10):
  #visualizing outputs for all the models for specific number of input images
  for i in range(start, start + test_img_count):
    x, y = gen.__getitem__(i)
    fig = plt.figure(figsize=(25,25))
    for idx, mod in enumerate(all_models):
      y_pred = mod.predict(x, verbose = 0)
      visualize_output(x.copy(), y, y_pred, test_img_count, idx+1, fig, i==start)
  plt.show()

def evaluate_all_models(gen, all_models, num_tests):
  #evaluating all models
  valid_ids = os.listdir(validation_path)
  gen = DataGen(valid_ids, "validation/", batch_size=1, image_size=image_size)
  num_tests = len(valid_ids)
  for idx, mod in enumerate(all_models):
    out = mod.evaluate_generator(gen, steps = num_tests)
    print(all_mods[idx], out[-1])

all_models = load_models() # pass any of the following as a string to load the specific model : 'half_unet', 'half_unet_reg', 'half_unet_norm', 'unet', 'nested_half_unet'
gen, num_tests = load_test_data(1) # pass any number higher than 1 which represents the batch_size for the data generator

evaluate_all_models(gen, all_models, num_tests)