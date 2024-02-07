import numpy as np
from pathlib import Path
from PIL import Image
import pickle

# This code needs an accompanying .pkl file to run, which is the pickled version of the trained machine learning model.
# Working along with the trained model this code will identify, with 85% accuracy, the image of a domestic power meter.
# A copy of the .pkl file (trained ML model) is available on my github along with the Jupyter Notebook code used to train this model.
# https://github.com/PrashantDas/Meter_Image_Recognition

# This function turns images into their corresponding array of pixels
def trim_image_to_square(file_path):    
    """the function crops the image to a square, then resizes it to 96 px X 96 px and returns a 1-d array of its pixels """
    img = Image.open(file_path)
    width = img.width
    height = img.height
    if width < height:
        crop_from_top = (height - width) // 2
        crop_from_bottom = crop_from_top + width
        squared_img = img.crop((0, crop_from_top, width, crop_from_bottom)).resize((96, 96))
        arr_data = np.array(squared_img)
    else:
        crop_from_left = (width - height) // 2
        crop_from_right = crop_from_left + height
        squared_img = img.crop((crop_from_left, 0, crop_from_right, height)).resize((96, 96))
        arr_data = np.array(squared_img) 
    return arr_data.flatten() / 255 # standardizing



############# User Input ###################
# Caveat: The model is trained on jpg images hence only jpg images can be tested

pickled_model_path = Path('C:/') / 'svc_meter_model.pkl'  # ammend the path as per your local drive
model = pickle.load(open(pickled_model_path, 'rb'))

test_image = Path('C:/') / '11348.jpg' # ammend the path as per your local drive
edited_image_array = trim_image_to_square(test_image)
prediction = model.predict(edited_image_array.reshape(1, -1))
prediction = int(prediction)
############# Output #######################

output_guide = {1: 'Image shows a power meter', 0: 'The image does not show a power meter'}
result = output_guide[prediction]
print('_'*70)
print(result.center(60, '*'))
print('_'*70)