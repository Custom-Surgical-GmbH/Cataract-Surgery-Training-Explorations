# Cataract Surgery Training Explorations

## Description

This project serves as an exploratory basis for other projects with a specific use in mind. The bulk of the work is contained in Jupyter notebooks. Some of the more concrete results are then rewritten in functions in the `opencv_explorations/helpers` directory. What has been tested until now is the following: 
- Tracking and detection of various Bioniko eye models
- Detection of microscope's view
- Outlier filtering methods
- Tracking smoothing methods
- Quantitative evaluation of various methods
- Automatic capsulorhexis segmentation

## Setup 

1. Download folder `data` from [Drive](https://drive.google.com/drive/folders/1yMcpuYanOFzhQJIz9-Kk2CYrbNeZBJm1?usp=sharing) and put it in the project path `opencv_explorations/data`
2. Create Python virtual environment and install all the requirements using the command `pip install -r requirements.txt`
3. Work on Jupyter notebooks directly from their directory as follows: `cd opencv_explorations && jupyter notebook .`


