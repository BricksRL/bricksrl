# PyBricksRL

## Enable web Bluetooth on chrome

 1. Go to "chrome://flags/" 
 2. enable "Experimental Web Platform features"
 3. restart chrome
 4. Use beta.pybricks.com to edit and upload the client scripts for each environment

## Setup Environment and Packages

Create your conda environment with: `conda create --name lego python=3.8`

Activate the conda environment with: `conda activate lego`

Install PyTorch: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Install packages needed with: `pip install -r requirements.txt`


## Client 
Update your client script on the [PyBricks Hub](https://beta.pybricks.com/) whenever you want to run a new environment


## Helpers:
We created a [PyBricks GPT](https://chat.openai.com/g/g-YZYA1j4yF-pybrics-lego-doc) which is an expert on the PyBricks documentation and can help you with coding and debugging client code for the PyBricks Hub.


# Additional Info
Code examples from PyBricks. Might be interesting as expert policies for offline RL. [GitHub Pybricks examples](https://github.com/pybricks/pybricks-projects/blob/master/sets/mindstorms-robot-inventor/main-models/gelo/gelo_basic.py)


