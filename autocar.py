import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'
from tensorflow.keras.models import load_model
import airsim
import numpy as np
import cv2

# Classe para controlar o carro com a rede neural
class AutoCar:
    
    def __init__(self, client, model, name='AutoCar'):
        self.client = client
        self.model = load_model(model)
        self.name = name

        self.client.enableApiControl(True, self.name)
        self.controls = airsim.CarControls()
        self.reset_controls()
    
    def reset_controls(self):
        self.controls.steering = 0
        self.controls.throttle = 0
        self.controls.brake = 0
        self.client.setCarControls(self.controls, self.name)
    
    def updateState(self):
        self.state = self.client.getCarState(self.name)
    
    def get_image(self, airsim_image_type):
        image_response = self.client.simGetImage("0", airsim_image_type, self.name)
        path_img = os.path.normpath('./img_nvidia_model_' + str(self.name) + '.png')
        airsim.write_file(path_img, image_response)

        transition_image = cv2.imread(path_img)
        transition_image = transition_image[60:143, 0:255]

        return np.array([transition_image])

    def race(self):
        ### === Predição === ###
        img = self.get_image(airsim.ImageType.Segmentation)
        model_output = self.model.predict(img)
        self.controls.steering = float(model_output[0][0])*2
        self.controls.throttle = 0.7 - (0.5*abs(self.controls.steering))
        self.client.setCarControls(self.controls, self.name)
        ### === Atualização sobre a pose do carro (race.py) === ###
        self.updateState()