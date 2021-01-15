import airsim
import os
import csv

### === Classe que gera um dataset em um ambiente AirSim. === ###
class airsim_dataset:

    def __init__(self, path):
        self.path = path
        self.images = []
        self.states = []
        self.it_imgs = 0

    def record(self, race, pid_car1, airsim_image_type):

        ### === Tira um foto da câmera frontal do carro. === ###
        image_response = race.client.simGetImage("0", airsim_image_type)
        self.images.append(image_response)

        ### === Pega o ângulo de direção (steering angle) e o nome da imagem de um estado do carro. === ###
        steering = f'{pid_car1.controls.steering:.6f}'
        img = 'img_' + str(self.it_imgs) + '.png'
        self.states.append([steering, img])
        self.it_imgs += 1
    
    ### === Salva os dados e cria o dataset. === ###
    def saving(self):
        print("Saving the dataset... {}".format(self.path))
        os.mkdir(self.path + 'images/')
        with open(self.path + 'airsim_rec.txt', 'wt', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Steering', 'ImageName'])
            for i in range(len(self.states)):
                tsv_writer.writerow(self.states[i])
                path_img = os.path.join(self.path, 'images/' + self.states[i][1])
                airsim.write_file(path_img, self.images[i])