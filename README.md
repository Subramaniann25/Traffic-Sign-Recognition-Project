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

Finally, we have stored all the images and their labels into lists (data and labels).

We need to convert the list into numpy arrays for feeding to the model.

The shape of data is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value).

With the sklearn package, we use the train_test_split() method to split training and testing data.

From the keras.utils package, we use to_categorical method to convert the labels present in y_train and t_test into one-hot encoding.

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/930f7a5e-a4ff-4701-ab77-12bd96047f26)

Step 2: Build a CNN model

To classify the images into their respective categories, we will build a CNN model (Convolutional Neural Network). CNN is best for image classification purposes.

The architecture of our model is:

2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
Flatten layer to squeeze the layers into 1 dimension
Dense Fully connected layer (256 nodes, activation=”relu”)
Dropout layer (rate=0.5)
Dense layer (43 nodes, activation=”softmax”)
We compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/b0d4ed5d-4aa7-4188-b87d-27a62856f3dd)

Steps 3: Train and validate the model

After building the model architecture, we then train the model using model.fit(). I tried with batch size 32 and 64. Our model performed better with 64 batch size. And after 15 epochs the accuracy was stable.

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/fca44aa8-7eec-4c15-939c-2a66558060e3)

Our model got a 95% accuracy on the training dataset. With matplotlib, we plot the graph for accuracy and the loss.
![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/d6ad376e-0aa0-4b43-996a-813ee7679b41)

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/40199628-0677-44ef-a08e-3860a4892eb8)

Step 4: Test our model with test dataset

Our dataset contains a test folder and in a test.csv file, we have the details related to the image path and their respective class labels. We extract the image path and labels using pandas. Then to predict the model, we have to resize our images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, we imported the accuracy_score and observed how our model predicted the actual labels. We achieved a 95% accuracy in this model.

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/5db9a9a4-a29b-4cf6-a73b-bafca375f012)

In the end, we are going to save the model that we have trained using the Keras model.save() function.

->  model.save(‘traffic_classifier.h5’)

OUTPUT:

![image](https://github.com/Subramaniann25/Traffic-Sign-Recognition-Project/assets/114677185/dd53b160-eef7-43cc-8184-dbdfa1dbe677)

Summary
In this Python project with source code, we have successfully classified the traffic signs classifier with 95% accuracy and also visualized how our accuracy and loss changes with time, which is pretty good from a simple CNN model.













