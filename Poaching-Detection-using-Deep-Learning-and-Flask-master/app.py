import os
from flask import Flask, render_template, request
import cv2
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import imghdr
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from twilio.rest import Client
import requests

# Custom loss function to handle 'auto' reduction
class CustomBinaryCrossentropy(BinaryCrossentropy):
    def __init__(self, **kwargs):
        if 'reduction' in kwargs and kwargs['reduction'] == 'auto':
            kwargs['reduction'] = 'none'  # or 'mean'
        super().__init__(**kwargs)

app = Flask(__name__)
target_folder = "/workspaces/Wildlife_Monitoring_Poaching_Preventation/images"

# Create the target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

def is_target_folder(folder_path, target_folder):
    folder_path = os.path.abspath(folder_path)
    target_folder = os.path.abspath(target_folder)
    return folder_path == target_folder

@app.route('/', methods=['GET', 'POST'])
def upload_folder():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if os.path.exists(folder_path):
            if is_target_folder(folder_path, target_folder):
                try:
                    # Load model with custom objects to handle 'auto' reduction
                    new_model1 = load_model(
                        os.path.join('models', 'poachingdetectionVER7.h5'),
                        custom_objects={'BinaryCrossentropy': CustomBinaryCrossentropy}
                    )
                    
                    poacher = False
                    person = int(0)
                    noperson = int(0)
                    solution = 0
                    cwd = os.getcwd()
                    print(cwd)
                    os.chdir(target_folder)
                    
                    # SMS INTEGRATION
                    response = requests.get("http://ip-api.com/json/").json()
                    message1 = " "
                    message1 = "The region of poaching is " + \
                        response['region']+" "+response['city'] + " latitude is " + \
                        str(response['lat']) + " logitude is "+str(response['lon'])
                    
                    cwd = os.getcwd()
                    print(cwd)
                    
                    # Check if there are any jpg files
                    jpg_files = [f for f in os.listdir() if f.endswith(".jpg")]
                    if not jpg_files:
                        return render_template('index.html', error="No jpg files found in the target folder.")
                    
                    for picture in jpg_files:
                        testingimg = cv2.imread(picture)
                        if testingimg is None:
                            print(f"Failed to load image: {picture}")
                            continue
                        
                        pic1 = tf.image.resize(testingimg, (256, 256))
                        solution = new_model1.predict(np.expand_dims(pic1, 0))
                        
                        if solution > 0.5:
                            print(f'poacher is present warning')
                            print(picture)
                            poacher = True
                            person = person+1
                        else:
                            print(f'No poacher is present')
                            print(picture)
                            noperson = noperson+1
                    
                    print(person)
                    print(noperson)
                    
                    finalmessage = True
                    if (person > (person+noperson)*.10):
                        finalmessage = True
                    else:
                        finalmessage = False
                    
                    print(finalmessage)
                    
                    if(finalmessage == True):
                        SID = ""
                        auth_token = ""
                        my_phone_number = ''
                        target_phone_number = ''
                        
                        # Only send SMS if credentials are provided
                        if SID and auth_token and my_phone_number and target_phone_number:
                            cl = Client(SID, auth_token)
                            if(poacher):
                                cl.messages.create(
                                    body=message1, from_=my_phone_number, to=target_phone_number)
                        
                        outputinscreen = "Poaching is present and SMS REGARDING POACHING is sent to concerned authorities"
                    else:
                        outputinscreen = "Poaching is not present and animals are safe"
                    
                    # Change back to original directory
                    os.chdir(cwd)
                    return render_template('index.html', prediction=outputinscreen)
                
                except Exception as e:
                    # Return to original directory in case of error
                    os.chdir(os.path.dirname(os.path.abspath(__file__)))
                    return render_template('index.html', error=f"Error processing images: {str(e)}")
            else:
                error = "Please use the correct target folder path: " + target_folder
                return render_template('index.html', error=error)
        else:
            error = "Invalid folder path. Please try again."
            return render_template('index.html', error=error)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)