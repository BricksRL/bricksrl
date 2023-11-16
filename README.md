# lego_robot

## Enable web Bluetooth on chrome

 1. Go to "chrome://flags/" 
 2. enable "Experimental Web Platform features"
 3. restart chrome
 4. Use beta.pybricks.com to edit and upload the hub program

## Install packages
```
conda create --name lego python=3.8
conda activate lego
pip install pybricksdev --pre
pip3 install torch torchvision torchaudio
pip3 install tensordict
pip3 install torchrl
pip3 install hydra-core
pip install wandb
```



# TODO:

- Milestone Cleanup []
    - update base env with changes in balance env. 
    - add balance env
    - cleanup base environment

- Milestone Offline Datasets [ ]
    - collect dataset for wall-distance/run away [ ]
    - create datasets that can be loaded for torchrl offline datasets [ ]
    - do offline training for run-away robot []
        - create script for offline training and evaluation [ ]

- Milestone CatRobot
    - integrate CatRobot environment
    - create CatRobot Env [ ]
    - create CatRobot offlline datasets []
    - other ideas but walking max distance? [ ]

- Milestone RoboArm 
    - integrate RoboArm environment [ ]
    - Possible task? 
        - reach some point e.g. ReachingEnv
        - Grab cube 
        - more ? 
    - Create offline dataset
    