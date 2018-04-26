import pandas as pd
import numpy as np
import time
script_start_time = time.time()

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from util import *

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.applications import VGG16, ResNet50, VGG19, InceptionResNetV2, DenseNet201, Xception, InceptionV3
from keras import backend as K
K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf') #channel last




def get_index_to_label(label_to_index):
    ensure_directory(processed_data_path)
    if not os.path.exists(os.path.join(processed_data_path, 'index_to_label.txt')):
        index_to_label = dict((v,k) for k,v in label_to_index.items()) #flip k,v
        import json
        with open(os.path.join(processed_data_path, 'index_to_label.txt'), 'w') as outfile:
            json.dump(index_to_label, outfile)


def add_new_layer(base_model, CLASS_SIZE):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) #new FC layer, random init
    x = Dropout(0.4)(x)
    predictions = Dense(CLASS_SIZE, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("{:<10} Pretrained model layers: {}".format('[INFO]', len(base_model.layers)))
    print("{:<10} New model layers       : {}".format('[INFO]', len(model.layers)))

def setup_to_finetune(model):
    layer_num = len(model.layers)
    trainable_layer_num = max(int(layer_num * 0.1), 8)
    for layer in model.layers[ : -trainable_layer_num]:
        layer.trainable = False
    for layer in model.layers[-trainable_layer_num : ]:
        layer.trainable = True
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    print("{:<10} Trainable model layers : {}".format('[INFO]', trainable_layer_num))
    print("{:<10} Unrainable model layers: {}".format('[INFO]', layer_num - trainable_layer_num))

def get_callbacks_list(train_choice):
    checkpoint = ModelCheckpoint('%s/%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5'%(model_path, train_choice), monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='auto',period=1)
    tensorboard = TensorBoard(log_dir="TensorBoard/logs/{}".format(time.time()))
    earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    callbacks_list = [earlystopping, checkpoint, tensorboard]
    return callbacks_list


def train():
    # data prep
    get_now()
    print("{:<10} Preparing data ...".format('[INFO]'))

    train_datagen =  ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    val_datagen = ImageDataGenerator(
        rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        image_path_train,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(
        image_path_val,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    get_index_to_label(train_generator.class_indices)


    print('%0.2f min: Start building, loading pretrained model %s'%((time.time() - script_start_time)/60, pretrained_model_name))
    base_model = pretrained_model(include_top=False, weights='imagenet')
    model = add_new_layer(base_model, CLASS_SIZE)
    setup_to_transfer_learn(model, base_model)
    print('%0.2f min: Loaded %s'%((time.time() - script_start_time)/60, pretrained_model_name))

    get_now()
    print('%0.2f min: Start building model %s'%((time.time() - script_start_time)/60, pretrained_model_name))
    build_start_time = time.time()
    model.fit_generator(
        train_generator,
        steps_per_epoch= TRAIN_SIZE // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = get_callbacks_list('build'),
        validation_data = val_generator,
        validation_steps= VAL_SIZE // BATCH_SIZE)
    model.save(os.path.join(model_path, 'build', pretrained_model_name+'.h5'))
    print('%0.2f min: Finish building model %s'%((time.time() - script_start_time)/60, pretrained_model_name))
    print("{:<10} It takes {:.2f} min to build the model".format('[INFO]', (time.time() - build_start_time)/60 ))

    get_now()
    print('%0.2f min: Start tuning model %s'%((time.time() - script_start_time)/60, pretrained_model_name))
    tune_start_time = time.time()
    setup_to_finetune(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch= TRAIN_SIZE // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = get_callbacks_list('tune'),
        validation_data = val_generator,
        validation_steps= VAL_SIZE // BATCH_SIZE)
    model.save(os.path.join(model_path, 'tune', pretrained_model_name+'.h5'))
    print('%0.2f min: Finish tuning model %s'%((time.time() - script_start_time)/60, pretrained_model_name))
    print("{:<10} It takes {:.2f} min to tune the model".format('[INFO]', (time.time() - tune_start_time)/60 ))


# Settings
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
url_data_path = os.path.join(data_path, 'urls')
image_path_train = os.path.join(data_path, 'ImagesTrain')
image_path_val = os.path.join(data_path, 'ImagesVal')
processed_data_path = os.path.join(data_path, 'ProcessedData')
model_path = os.path.join(data_path, 'Model')
image_path_predict = os.path.join(project_path, 'Predict')

classes = [f for f in get_subfolder_names(image_path_train)]
sub_folders_image_counts_train = [len(get_sub_fnames(cate_folder)) for cate_folder in get_subfolder_paths(image_path_train)]
sub_folders_image_counts_val = [len(get_sub_fnames(cate_folder)) for cate_folder in get_subfolder_paths(image_path_val)]



pretrained_models = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'InceptionResNetV2': InceptionResNetV2,
    'VGG19': VGG19,
    'DenseNet201': DenseNet201,
    'Xception': Xception,
    'InceptionV3': InceptionV3}

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Training models')
    argument_parser.add_argument('--model', type=str, default='VGG16', help='The pretrained image classification model - - VGG16, ResNet50, VGG19, InceptionResNetV2, DenseNet201, Xception or InceptionV3')
    argument_parser.add_argument('--epoch', type=str, default= 25, help='EPOCH for training the model. 25 by default')
    args = argument_parser.parse_args()

    pretrained_model_name = args.model
    pretrained_model = pretrained_models[pretrained_model_name]
    EPOCHS = args.epoch

    SEED = 0
    BATCH_SIZE = 32
    EPOCHS = 25
    IM_WIDTH, IM_HEIGHT = 150, 150
    CLASS_SIZE = len(classes)
    TRAIN_SIZE = sum(sub_folders_image_counts_train)
    VAL_SIZE = sum(sub_folders_image_counts_val)

    get_now()
    print("{:<10} Numbe of Classes: {}".format('[INFO]', len(classes)))
    print("{:<10} Classes: {}".format('[INFO]', str(classes)))
    print("{:<10} Train Size: {}".format('[INFO]', str(sub_folders_image_counts_train)))
    print("{:<10} Val Size: {}".format('[INFO]', str(sub_folders_image_counts_val)))
    print("{:<10} epoch: {}".format('[INFO]', EPOCHS))
    print("{:<10} batch size: {}".format('[INFO]', BATCH_SIZE))

    ensure_directory(model_path)
    model_path = os.path.join(model_path, pretrained_model_name)
    ensure_directory(model_path)
    ensure_directory(os.path.join(model_path, 'build'))
    ensure_directory(os.path.join(model_path, 'tune'))
    print("{:<10} Pretrained Model: {}".format('[INFO]', pretrained_model_name))


    train()
    get_now()
    print("{:<10} It takes {:.2f} min to finish training your model".format('[CONGRATS]', (time.time() - script_start_time)/60 ))
    sys.exit(1)
