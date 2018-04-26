import numpy as np
import os, sys
import argparse
from util import *

import warnings
warnings.filterwarnings('ignore')


def process_single_image_for_predict(img, target_size):
    from PIL import Image
    img = img.resize(target_size,Image.ANTIALIAS)
    from keras.preprocessing.image import img_to_array
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img  = img.astype('float32')
    img /= 255
    return img

def getLabel(prob):
    import json
    with open('%s/index_to_label.txt'%(processed_data_path)) as json_data:
        index_to_label = json.load(json_data)
        json_data.close()

    index_to_label = dict((int(k), v) for k,v in index_to_label.items())
    prediction_index = np.argmax(prob, axis=-1) #multiple categories
    prediction_label = [index_to_label[k] for k in prediction_index]
    return prediction_label

def get_ordered_label():
    import json
    with open('%s/index_to_label.txt'%(processed_data_path)) as json_data:
        index_to_label = json.load(json_data)
        json_data.close()

    index_to_label = dict((int(k), v) for k,v in index_to_label.items())
    return [index_to_label[key] for key in sorted(index_to_label.keys())]



# Settings
IM_WIDTH, IM_HEIGHT = 150, 150
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
image_path_train = os.path.join(data_path, 'ImagesTrain')
image_path_val = os.path.join(data_path, 'ImagesVal')
processed_data_path = os.path.join(data_path, 'ProcessedData')
model_path = os.path.join(data_path, 'Model')
image_path_predict = os.path.join(project_path, 'Predict')

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--image_path", help="path to image")
    a.add_argument("--image_url", help="url to image")
    a.add_argument("--model")
    args = a.parse_args()

    if args.image_path is None and args.image_url is None:
        a.print_help()
        sys.exit(1)

    if args.model is None:
        model_choice = 'VGG16'
        print("{:<10} The default pretrained model will be used: {}".format('[INFO]', model_choice))
        print("{:<10} You can selecting model by adding argument - VGG16, ResNet50, VGG19, InceptionResNetV2, DenseNet201, Xception or InceptionV3 {}".format('', model_choice))
    else:
        model_choice = args.model

    # Load model
    print("{:<10} Start loading model".format('[INFO]', model_choice))
    from keras.models import load_model
    tuned_model_path = os.path.join(model_path, model_choice, 'tune', model_choice+'.h5')
    try:
        model = load_model(tuned_model_path)
    except:
        print("{:<10} Cannot load model trained from {}".format('[ERROR]', model_choice))
        exit(1)

    # Load image
    print("{:<10} Start loading image".format('[INFO]', model_choice))
    if args.image_path is not None:
        img = read_image_from_path(args.image_path)
    else:
        img = read_image_from_url(args.image_url)

    print("{:<10} Start Predicting".format('[INFO]', model_choice))
    prob = model.predict(process_single_image_for_predict(img, (IM_WIDTH, IM_HEIGHT)))
    predict_label = getLabel(prob)[0]
    ordered_labels = get_ordered_label()

    import random
    label_image_path = random.choice(get_sub_fpaths(os.path.join(image_path_train, predict_label)))
    label_image = read_image_from_path(label_image_path)

    print("{:<10} It is a {}".format('[RESULT]', predict_label.upper()))
    display_two_images([img, label_image], text = 'Similarity with %s: %.2f'%(predict_label.upper(), max(prob[0])))
    plot_prob_radar(prob[0], ordered_labels, title = 'Similarity')

    sys.exit(1)
