"""
    Código de treinamento de uma rede neural (baseado no modelo da NVIDIA - End-To-End Deep Learning - AutoPilot ConvNet)

    O principal objetivo dessa rede neural é realizar a predição de um ângulo de direção (steering angle) de um carro autônomo
    no ambiente simulado, Microsoft AirSim. A técnica aplicada aqui se chama 'Behavioral Cloning'.
"""

### Obs.: Quando encontrar: << ... >>. Significa que você é orientado a mudar se for necessário. ###

### === Módulos/Pacotes === ###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'
import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam

### === Construção do Dataset === ###
# << Diretório do dataset padrão >>
MAIN_DATA_DIR = "./raw_data/run-center-surf/"

# << Leitura do arquivo log dos estados do carro >> (O formato padrão do arquivo é '.tsv' - tab separated values).
LOG_FILENAME = MAIN_DATA_DIR + "airsim_rec.txt"
print("Reading... {}".format(LOG_FILENAME))
data = pd.read_csv(LOG_FILENAME, sep="\t")
print("Done!", end="\n\n")

# << Subdivide o arquivo log em 2 campos do nosso interesse. >>
steering = data["Steering"]
img_files = data["ImageName"]

# << Coleta todas as imagens, e aplica um certo ROI. >>
print("Loading the images...")
images=[]
for image in data["ImageName"]:
    transition_image = cv2.imread(MAIN_DATA_DIR + "images/" + image)
    transition_image = transition_image[60:143,0:255] 
    images.append(transition_image)
print("Done!", end="\n\n")

# Definindo o dataset para rede neural.
X_train = np.array(images)
y_train = np.array(steering)

input_shape = X_train.shape[1:]

### === Setup da rede neural === ###
# << Número de épocas. >>
num_epochs = 200

# << Limite do número de épocas estagnadas no mesmo custo, atingindo esse limite o treinamento para. >>
trainning_patience = 20

# << Taxa de aprendizado para o método do gradiente >>
learning_rate = 0.0001

# << Determina o tamanho(%) do conjunto de avaliação. >>
validation_split = 0.2

# << Tamanho do lote de exemplos >>
batch_size = 64

# << Diretório onde vai ser salvo os modelos neurais. >>
MODEL_OUTPUT_DIR = "./nvidia_model/"
os.mkdir(MODEL_OUTPUT_DIR)

### === Rede neural convolucional === ###
print("Creating the CNN...")
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))

model.add(Conv2D(24, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(36, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(48, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='tanh'))
print("Done!", end="\n\n")

# Compilação da rede neural com otimizador do método do gradiente: "Adam".
print("Compiling the CNN...")
adam = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
print("Done!", end="\n\n")

# Metadados da rede neural.
model.summary()

# << Callbacks que supervisionam o treinamento. >>
plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}-{2}.h5'.format('nvidia', '{epoch:02d}', '{val_loss:.7f}'))
checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=trainning_patience, verbose=1)
callbacks=[plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]

# Treinamento da rede neural.
print("Trainning...")
historic=model.fit(X_train, y_train, validation_split=validation_split, shuffle=True, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
print("Done!", end="\n\n")

### === Estatística do custo e eficiência da rede neural === ###
plt.plot(historic.history['loss'])
plt.plot(historic.history['val_loss'])
plt.title('Model Mean Squared Error')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Trainning Set', 'Validation Set'], loc='upper right')
plt.show()