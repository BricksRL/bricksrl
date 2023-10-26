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

- update base env with hcanges in balance env. 
- add balance env 
- run with current changes runaway
- cleanup base environment 
