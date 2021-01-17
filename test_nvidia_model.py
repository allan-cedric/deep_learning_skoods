"""
    Código para testar a rede neural criada em 'train_nvidia_model.py' no ambiente AirSim.
"""

### === Módulos/Pacotes === ###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import time
import sys
import numpy as np
import glob
import airsim

### === Tratamento da imagem da câmera === ###
def get_image(airsim_image_type):
    image_response = client.simGetImage("0", airsim_image_type)
    path_img = os.path.normpath('./img_nvidia_model.png')
    airsim.write_file(path_img, image_response)

    transition_image = cv2.imread(path_img)
    transition_image = transition_image[60:143, 0:255]

    return np.array([transition_image])

### === Carrega o modelo de menor custo === ###
models = glob.glob('./nvidia_model-run-center-surf/models/*.h5')
best_model = max(models, key=os.path.getctime)
MODEL_PATH = best_model
print('Using model {0} for testing.'.format(MODEL_PATH))
model = load_model(MODEL_PATH)

### === Conexão com o ambiente Airsim === ###
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connection established!')

### === Setup inicial do carro === ###
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
client.setCarControls(car_controls)

### === Programa principal === ###
while True:

    ### === Predição do modelo neural === ###
    img = get_image(airsim.ImageType.SurfaceNormals)
    model_output = model.predict(img)
    
    ### === Controle do carro (nvidia_mode-run-center-surf1) === ###
    '''
    car_controls.steering = float(model_output[0][0])*2
    car_controls.throttle = 0.7 - (0.4*abs(car_controls.steering))'''
    car_controls.steering = float(model_output[0][0])*2
    car_controls.throttle = 0.65 - (0.35*abs(car_controls.steering))

    ### === Imprime no console o ângulo de direção (steering angle) e a intensidade do acelerador (throttle) === ###
    # print('Sending steering = {0}, Sending throttle = {1}'.format(car_controls.steering, car_controls.throttle))

    ### === Atualize os controles do carro === ###
    client.setCarControls(car_controls)

    ### === Aguarda um tempinho antes da próxima iteração === ###
    # time.sleep(0.01)

client.enableApiControl(False)
