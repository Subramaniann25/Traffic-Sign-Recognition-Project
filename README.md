# Traffic-Sign-Recognition-Project
It is necessary for the vehicle to understand and follow the traffic rules. For achieving accuracy in this technology the vehicles should be able to interpret traffic signs.

There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.

In this Python project example, we will build a deep neural network model that can classify traffic signs present in the image into different categories. With this model, we are able to read and understand traffic signs which are a very important task for all autonomous vehicles.

The Dataset of Python Project
For this project, we are using the public dataset available at Kaggle:

Traffic Signs Dataset

The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The size of the dataset is around 300 MB. The dataset has a train folder which contains images inside each class and a test folder which we will use for testing our model.

Prerequisites
This project requires prior knowledge of Keras, Matplotlib, Scikit-learn, Pandas, PIL and image classification.

To install the necessary packages used for this Python data science project, enter the below command in your terminal:

->  pip install tensorflow keras sklearn matplotlib pandas pil

Create a Python script file and name it traffic_signs.py in the project folder.

Our approach to building this traffic sign classification model is discussed in four steps:

Explore the dataset
Build a CNN model
Train and validate the model
Test the model with test dataset
Step 1: Explore the dataset

Our ‘train’ folder contains 43 folders each representing a different class. The range of the folder is from 0 to 42. With the help of the OS module, we iterate over all the classes and append images and their respective labels in the data and labels list.

The PIL library is used to open image content into an array

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/cdd79a1f-4c29-47e8-8c97-29c0b18e293e)

