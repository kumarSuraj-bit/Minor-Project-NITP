import argparse
import numpy as np
import cv2
from keras.models import load_model     #load our saved model

import base64   #decoding camera images


import socketio     #real-time server
import eventlet     #concurrent networking 
import eventlet.wsgi    #web server gateway interface
from flask import Flask     #web framework


from PIL import Image   #image manipulation
from io import BytesIO      #input output





#initialize our server
sio = socketio.Server()

#flask (web) app
app = Flask(__name__)


MAX_SPEED = 18
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED


def Preprocess(image):
    """
    image Processing
    """
    #print("we are in preprocess")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # Convert the image from RGB to YUV (This is what the NVIDIA model does)
    #image = cv2.resize(image, (320, 80))
    image = image/255 # normalize pixels value between 0 & 1
    return image

@sio.event
def connect_error(data):
    print("The connection failed!")

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])

        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])

        # The current speed of the car
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))


        try:
            
            image = np.asarray(image)       # from PIL image to numpy array
            image = Preprocess(image)       # apply the preprocessing
            image = np.array([image])       # the model expects 4D array
            # predict the steering angle for the image
            #print(image.shape)
            
            steering_angle = float(model.predict(image, batch_size=1))
            #steering_angle = 0.2

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
            
        except Exception as e:
            print(e)

    else:
        print("we are not in telemetry")
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(   
                "steer",
                data={
                        'steering_angle': steering_angle.__str__(),
                        'throttle': throttle.__str__()
                    },
                skip_sid=True
            )



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automated car Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model file. Model should be on the same path.'
    )
    
    args = parser.parse_args()

    #load model
    model = load_model(args.model)

    print("Server has started for autonomus car driving")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)