from setuptools import find_packages, setup

setup(
    name="bricksrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pybricksdev",
        "tensordict==0.5.0",
        "torchrl==0.5.0",
        "hydra-core==1.3.2",
        "wandb==0.16.1",
        "opencv-python==4.9.0.80",
        "moviepy==1.0.3",
        "tqdm==4.66.1",
        "numpy==1.24.1",
        "pynput",
    ],
    extras_require={
        "dev": [
            "pytest==8.0.2",
            "ufmt",
            "pre-commit",
        ],
    },
    author="Sebastian Dittert",
    description="BricksRL: A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO",
    url="https://github.com/BricksRL/bricksrl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
