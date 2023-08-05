# WavFileSteganographyCS4463

## Summary
Welcome to Group 8's program! We have trained a machine learning model to detect if a WAV file
has another file hidden in varrying degrees of the Least Significant Bit (1 through 8).

We implemented a research paper called "Detecting information-hiding in WAV audios", which can
be found here:

https://ieeexplore.ieee.org/document/4761650

## Accessing from .zip
The executable file (wavfilestego.exe) can be found directly in the wavfilestego directory. 

## Execution
The executable was created with PyInstaller. 

No command line arguments are required for the program. There are 9 training folders and 9
testing folders that contain WAV files. There is a .csv file for each folder that labels the file
names with a 0 for no files hidden and 1 for a hidden file. 

## Menu Options
You navigate the program through a menu interface. 

### 1. List Available Files for Steganalysis
You can see which files the program is able to perform a steganalysis on. All of these files are
in a folder called steganalysis_files. If you wish to add more files for steganalysis, please place a WAV file in that folder. 

### 2. Enter LSB Value for Training
Choose a number 1 through 9 for the training data. Numbers 1 through 8 train with files of that
degree of LSB hiding. 9 does uses a training folder that contains files that use all 8 degrees of LSB hiding. If you choose to press "q" and quit to the main menu, 9 will be selected by default. If you know which degree of LSB hiding is used on a file you are testing, it is recommended you choose that option. All WAV were hidden by the StegAudio32_v1.01exe program provided to us.

### 3. Select File to Analyze
You are given the option to input a file or directory to be directly analyzed by the program. A WAV file must be detected in order to be used for input. If you choose to press "q" and quit to the main menu, the entire steganalysis_files folder will be selected by default. If you want to you can choose an individual file. If the file is in a folder, the path must be included. Example:
steganalysis_files/Test.wav

### Train and Test Model
You must enter the LSB value for training before reaching this step. If you try to train and test the model first, it will take you back to the main menu. It will take a few moments, but the program will output a list of predictions about if a file is hidden or not as well as the degree of confidence (accuracy) from the program. If you chose option 9 (the default) from the LSB value selection, the program may take a few minutes to produce output. 

### Exit Program 
This exits the program.

## Accuracy (as of August 4th, 2023)
Most of the time the program will output an accuracy of approximately 40-60%. We believe we can tune the accuracy more by adding more variety to our training. This can be done by manipulating files included in the folders. No editing of code should be needed. 

## Fun Fact
It's a ".wav" file, not a ".wave" file. 