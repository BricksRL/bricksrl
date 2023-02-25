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
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
