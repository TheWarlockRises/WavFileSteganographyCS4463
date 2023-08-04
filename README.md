# WavFileSteganographyCS4463

## Summary
Welcome to Group 8's program! We have trained a machine learning model to detect if a WAV file
has another file hidden in varrying degrees of the Least Significant Bit (1 through 8).

We implemented a research paper called "Detecting information-hiding in WAV audios", which can
be found here:

https://ieeexplore.ieee.org/document/4761650

## Accessing from Github
The executable file (wavfilestego.exe) can be found in WavFileSteganographyCS4463/dist/wavfilestego. 

## Accessing from .zip
The executable file (wavfilestego.exe) can be found directly in the wavfilestego directory. 

## Execution
The executable was created with PyInstaller. 

No command line arguments are required for the program. There are 9 training folders and 9
testing folders that contain WAV files. There is a .csv file for each folder that labels the file
names with a 0 for no files hidden and 1 for a hidden file. 

## Menu Options
You navigate the program through a menu interface. 

### 1. List Files
You can see which files the program can use.

### 2. Analyze Files

### 3. Exit Program
Inputting "3" will exit the program. 

## Fun Fact
It's a ".wav" file, not a ".wave file. 