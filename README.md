# Face Recognition Attandance System

### Recognize The faces And Take Automatic Attandance. :sparkles:

![Face Recognition Logo](https://github.com/kmhmubin/Face-Recognition-Attendance-System/blob/master/Document%20Metarial/Project%20demo%20images/Face-Recognition-Attendance-System-Logo.jpg)


![GitHub](https://img.shields.io/github/license/kmhmubin/Face-Recognition-Attendance-System)

## Motivation :astonished:
----------------------------
We seek to provide a valuable attendance service for both teachers and students. Reduce manual process errors by provide automated and a reliable attendance system uses face recognition technology.

## Features :clipboard:
---------------------------
* Check Camera
* Capture Faces
* Train Faces
* Recognize Faces & Attendance
* Automatic Email

## Screenshots :camera:
-----------------------------------
### Command Line Interface

![Command Line Interdace](https://github.com/kmhmubin/Face-Recognition-Attendance-System/blob/master/Document%20Metarial/Project%20demo%20images/CODE%20INTERFACE.png)

### Checking Camera

![Checking Camera](https://github.com/kmhmubin/Face-Recognition-Attendance-System/blob/master/Document%20Metarial/Project%20demo%20images/Program%20working.jpg)

### Automail 

![Automail](https://github.com/kmhmubin/Face-Recognition-Attendance-System/blob/master/Document%20Metarial/Project%20demo%20images/automail.jpg)


## Tech Used :computer:
--------------------------
Build With - 
* Python 3.7

Module Used -

All The Module are Latest Version.
* [OpenCV](https://docs.opencv.org/3.4/index.html) (Contrib 4.0.1)
* [Pillow](https://pypi.org/project/Pillow/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Shutil](https://docs.python.org/3/library/shutil.html)
* [CSV](https://docs.python.org/3/library/csv.html)
* [yagmail](https://pypi.org/project/yagmail/)


Face Recognition Algorithms -
* [Haar Cascade](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
* [LBPH (Local Binary Pattern Histogram)](https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html)

Software Used -
* [Pycharm 2019.2](https://www.jetbrains.com/pycharm/download/?section=windows)
* [VS CODE](https://code.visualstudio.com/download)
* [Git](https://git-scm.com/downloads)

## Installation :key:
-----------------------------------

#### Download or Clone the project

First Download or Clone the Project on Your Local Machine.To download the project from github press **Download Zip**

![Download Zip]

or 

You can clone the project with git bash.To clone the project using git bash first open the git bash and write the following code
```
git clone https://github.com/Ed-Gigaversity/Face-Recognition-Attendance-System.git
```
demo 

After download, Open the project using **Pycharm or VSCODE**. Then we have to create an python enviroment to run the program.

#### create enviroment 
First open the terminal or command line in the IDE.Then write the following code.
```
python -m venv env
```
Then activate the enviroment using the code below for windows.
```
.\env\Scripts\activate
```
[ *Notice:*
If your pc don't have virtual enviroment or pip install the follow this link.
[How to create Virtual Enviroment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) ]

#### Installing the packages
--------------------------------------------------

After creating the enviroment on your project let's install the necessary packages. 

To install those package open the terminal or command line and paste the code from below

```
pip install opencv-contrib-python
```
```
pip install numpy
```
```
pip install pandas
```
```
pip install Pillow
```
```
pip install pytest-shutil
```
```
pip install python-csv
```
```
pip install yagmail
```

[ **Notice: During the package installization, sometime it shows some error, to avoid those error you can install those packages as admin. ]

## Test Run :bicyclist:
-----------------------
After creating the enviroment and installing the packages, open the IDE terminal/command line to run the program. Using the code below.

```
py main.py
```
Here is a demo to run the program. I'm Using the Pycharm IDE in my demo.

![Test Run](https://github.com/kmhmubin/Face-Recognition-Attendance-System/blob/master/Document%20Metarial/Project%20demo%20images/code%20demo_edit_0.gif)

## How To Use? :pencil:
----------------------
If you want to use it just follow the steps below.

1. First download or clone the project
2. Import the project to your favourit IDE
3. Create an python enviroment
4. Install all the packages 
5. Change the mail information
6. Run the project using the command line or your IDE Run Button

## Known Bugs :bug:
------------------------------
This project have some bugs.

* <strike>Student Details: In student details folder the **StudentDetails.csv** file don't have ID & name column.This problem show when the program run first time and create the <stong>StudentDetails.csv</strong> file automatically. To soleve the problelm just open the file and add *ID & Name Column* in the file and save it.</strike>
* Auto Attachment: This is not a problem actually. The problem is before sent auto mail we have to manually change the file name. I tried to automate the attachment but i faild.
