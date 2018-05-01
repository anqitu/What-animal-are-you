# What-animal-are-you

Code to recognize what animal are you and help you train self-built simple machine learning model to analyze what animal do you look like.

## Basic Level:
The current model I have trained can only recognize 7 animals - 'Tiger', 'Cat', 'Koala', 'Rabbit', 'Dog', 'Lion', 'Fox'.
If you are doubtful about the model's correctness, you can use it to analyze pictures of real animals to test its accuracy.

You can try predicting a picture on local storage by doing:
```
python 3_Predct_Single_Image.py --image_path single/image/path.jpg  (For Single Image)
python 3_Predct_Single_Image.py --image_path image/folder/path  (For Batch Images from a folder)
python 3_Predct_Single_Image.py --image_url image_url  (url of an online image)
```

Or predicting a picture from online:
```
python 3_Predct_Single_Image.py --image_url image_url
```

Here are some output examples
![PredictAnimal](https://github.com/anqitu/What-animal-are-you/blob/master/PredictResult/Animal/20180426-225408-Lovely%20Rabbit.jpg)

![PredictHuman](https://github.com/anqitu/What-animal-are-you/blob/master/PredictResult/Human/20180426-222028-Lovely%20Rabbit.jpg)

## Advanced Level
If you are not satisfied with the current limited animal categories, or not satified with the model's accuracy, you can train your own model. All you need is to provide the categories' labels (and some time for training a new model).

You can try building a model yourself with your categories defined by yourself by running the scripts one by one:
```
python 0_Extract_Image_url.py --query cat --count 500 --label 'loyal dog'
python 0_Extract_Image_url.py --query cat --count 500 --label 'curious cat'
python 0_Extract_Image_url.py --query cat --count 500 --label 'cute koala'
python 1_image_Downloader.py --url_fpath all --count 100
python 2_Train_Model.py --model VGG16 --epoch 25
python 3_Recognize_Image.py --image_path image/path.jpg
```

## Notes:
  - 0_Extract_Image_url.py extracts urls for image of specified query from Google Image Search and save the urls into the Datafile/urls foldel.
  - 1_image_Downloader.py downloads images from the url into the Datafile/ImagesTrain and Datafile/ImagesVal
  - 2_Train_Model.py trains a model. This might take a some time, depending on the training data size.
  - The codes will automatically create required subfolders in the current project directoryself. So, you just need to make sure the Datafile/urls subfolder is inside the DataFile folder before running the script.
  - Some required dependencies - keras, numpy, matplotlib, selenium ...
