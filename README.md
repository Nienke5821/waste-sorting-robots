# Waste Sorting Robots

## About the project
According to CBS, Dutch municipalities collected 8.1 billion kilograms of household waste in 2023. The amount of PMD waste, Plastic (bottles), metal packaging and beverage cartons, per person was converted 21 kilograms (Centraal Bureau voor de Statistiek, 2024). All this waste needs to be collected, cleaned and sorted. These processes are dependent on large machine parks. However, to reduce this dependence there is a growing need for mobile robotic solutions that can navigate through complex environments, such as unstructured waste piles.

The goal of this project is to illustrate the research done for each capability (detection, locomotion, manipulation). As proof of concept, for each capability this project focuses exclusively on plastic water bottles, a representative plastic waste category with consistent shape but common real-world deformations. Therefore, the detection part will focus on finding the best algorithm to detect plastic bottles, while the locomotion part will focus on training a robot to move towards a bottle. Finally, the manipulation will look at how to decide the best orientation of the gripper, so that the gripper can grab plastic waste, in this case a plastic bottle.

## Usage
Our project can be divided into three parts, as can also be read in the report. These are:
- Detection: detecting a plastic bottle in the environment
- Locomotion: moving towards the detected bottle
- Manipulation: orienting the robot claw such that it can pick the bottle up
### Detection
The Detection folder contains scripts and resources related to plastic bottle detection using YOLO. It contains the following components:

Preprocessing:

"preprocessing.py":  Filters out all non-plastic bottle classes from the online dataset.

Training: 

"train_strat_A&B.py": Trains models using two strategies — (A) the existing online dataset and (B) the augmented version of the online dataset.

"train_strat_C.ipynb": Trains a model using a synthetic dataset (strategy C). It also includes training with a subset of the online dataset and compares its performance to the synthetic-only model.

"config_WARP.yaml", "config_WL.yaml": Specifies the existing online datasets used in the training.

Training results are saved in the "results_YOLO" folder.

Test: 

"test.py": Defines the evaluation procedure for testing multiple models on datasets derived from the original online dataset.

"config_test.yaml": Specifies the test datasets used in the evaluation.

"yolo_on_testdata.py": 

"yolo_on_realdata.py": 

Real-world test videos are located in the "testdata" folder.

Results of real-time detection using the Quadruped Robot with a mounted ZED2 camera are available in the "results_quadruped_detection" folder.



### Locomotion

### Manipulation
The Manipulation folder contains three python files. "compare methods.py" is used to find the best way to compute a line through a straight part of the bottle. This compiles 5 images, also saved in the Orientation folder.

"SAM2 live.py" is the final algorithm that can be used to get the directions for the robot claw to move. It shows you the live input of the camera, together with the drawn outline and line through the bottle if detected. It also displays the movement instructions on the screen.

"Sam2 testing.py" is the algorithm used for the results section of the report. In here, the movement directions are based on moving the bottle instead of the camera. This is since this is easier when using a webcam, and theoretically the results are the same.

The results of the tests for the Manipulation part are recorded using this testing file and can also be seen in this folder. There are four videos:
- one bottle.avi: recording of one not crumbled bottle
- two bottles.avi: recording of two not crumbled bottles
- one crumbled bottle.avi: recording of a crumbled bottle
- two crumbled bottles.avi: recording of two crumbed bottles

## How to run

### Detection
You can run the codes by running the files in a terminal. Keep in mind that the paths in both the Python files as the corresponding .yaml files are in line with your own file structure. Keep in mind that it is not possible to change the names of the directories that are still in the files of this GitHub page, because the structure is necessary if you are using YOLOv9 from Ultralytics.
### Locomotion

### Manipulation
You can run the "compare methods" file by running the file in the terminal. You can run the "SAM2 live" file also by running the file in the terminal. For this one, make sure there is a camera input. This is the same for the "SAM2 testing" file. Make sure you are located in the Manipulation folder when running, since it needs other files from this folder.

## Authors
Britt Erkens (1468359) 

Dan Wu (2176963)

Fleur Bouwman (2204452)

Nienke Dekkema (2080958)
