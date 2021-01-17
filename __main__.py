##########################################
### Skoods.org -> Self-Racing Car Team ###
##########################################

import airsim
import time
from skoods import race
from garage import pid_car

# To record the dataset
from airsim_dataset import airsim_dataset

# To use a Neural Network
import os
import glob
from autocar import AutoCar

# Setup for waypoints and dataset
dataset_dir = './raw_data/run-fast4-surf/'
filename_pid = 'run-fast4.pickle'

# Connect to Skoods simulation
sample_time = 0.01  # Define the sample time to perform all processing.
race = race.Race(sample_time)

# INITIALIZE CARS

# OPTION A: Qualify, record waypoints or record a dataset
# Need to change the settings.json file. Check the JSON_examples folder
'''
pid_car1 = pid_car.Car(race.client, race.sample_time, 'AutoCar', race.mode_input, filename=filename_pid)  # Give the car the name you want
cars = [pid_car1]'''


# OPTION B: Race 3 cars
# Need to change the settings.json file. Check the JSON_examples folder
'''
pid_car1 = pid_car.Car(race.client, race.sample_time, 'SetCarName1', race.mode_input, filename='run-fast4.pickle') # Give the car the name you want
pid_car2 = pid_car.Car(race.client, race.sample_time, 'SetCarName2', race.mode_input, waypoints_correction=[0, -7], filename='run-fast2.pickle') 
pid_car3 = pid_car.Car(race.client, race.sample_time, 'SetCarName3', race.mode_input, waypoints_correction=[0, -14], filename='run-fast4.pickle')
cars = [pid_car1, pid_car2, pid_car3]
'''

# OPTION C: Neural Network
### === Load the best neural model === ### (Take some time to find)
models = glob.glob('./nvidia_model-run-fast4-surf/models/*.h5')
best_model = max(models, key=os.path.getctime)
MODEL_PATH = best_model
auto_car1 = AutoCar(client=race.client, model=MODEL_PATH, name='AutoCar')
# auto_car2 = AutoCar(client=race.client, model=MODEL_PATH, name='AutoCar2')
cars = [auto_car1]

if race.mode_input == '1':  # Record Waypoints
    # Will run only the first car to record waypoints. Change settings.json file to only one car.
    cars[0].recordWaypointsToFile()

elif race.mode_input in ['2', '3', '4']:
    # Will run only the first car to Qualify. Change settings.json file to only one car.
    if race.mode_input == '2':
        race.setNumberOfLaps(3)
        cars = [cars[0]]
    elif race.mode_input == '3':
        race.setNumberOfLaps(3)
    # Record a dataset (it needs a waypoints file)
    elif race.mode_input == '4':
        race.setNumberOfLaps(18)
        cars = [cars[0]]
    
    race.setCars(cars)    
    race.setInitialTime()
    keep_racing = True
    if race.mode_input in ['2', '3']:
        while(keep_racing):
            for each_car in cars:
                # RUN YOUR CODE HERE
                # keep_racing_from_car not being used, but I will leave here just in case
                keep_racing_from_car = each_car.race()
                # END HERE3
            race.playSimulation()  # Will check for mode
            keep_racing_from_race = race.updateRaceParameters()
            # you can add more interruptions if needed
            keep_racing = (keep_racing_from_car and keep_racing_from_race)
        race.client.reset()
        for each_car in cars:
            race.client.enableApiControl(False, each_car.name)
    elif race.mode_input == '4':
        dataset = airsim_dataset(dataset_dir)
        while(keep_racing):
            for each_car in cars:
                # RUN YOUR CODE HERE
                # keep_racing_from_car not being used, but I will leave here just in case
                keep_racing_from_car = each_car.race()
                dataset.record(race, pid_car1, airsim.ImageType.SurfaceNormals)
                # END HERE3
            race.playSimulation()  # Will check for mode
            keep_racing_from_race = race.updateRaceParameters()
            # you can add more interruptions if needed
            keep_racing = (keep_racing_from_car and keep_racing_from_race)
        race.client.reset()
        for each_car in cars:
            race.client.enableApiControl(False, each_car.name)
        dataset.saving()
