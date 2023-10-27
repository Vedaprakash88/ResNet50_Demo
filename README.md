# ResNet50_Demo
Transfer Learning with ResNet50


ResNet50 knowledge can be transferred to a target job using multiple techniques. The pre-trained model is utilized to begin the new task, and the model's weights are updated to fit the new dataset. For the purpose of this report, the pre-trained model has been used as feature extractor in current model that feeds the output to a bespoke classifier (i.e., fully connected layers). The steps involved in this are discussed below:
1.5.1.	Pre-trained ResNet50 Model Selection:
Most of the pretrained models are readily available in deep learning frameworks such as TensorFlow/Keras or PyTorch. For the purposes of this report, we have selected the ResNet50 model, which has been pre-trained on a source task (i.e., the ImageNet database) using Keras, as depicted in Figure 4. The code is described below:

 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/3d6636fe-d1e1-49ad-9220-bd7887e44523)

Figure 4: Pre-Selection of ResNet50 model
•	The code uses TensorFlow (tf) to create a neural network model. tf version 2.14.0 is used for the purpose of this report.
•	The number of classes are set to 4, as this model is designed for a classification task with four classes.
•	A new sequential model named ‘my_model’ is created using tf.keras.Sequential().
•	The ResNet50 model from TensorFlow-Kera's applications module is added to ‘my_model’.
o	the top layer (fully connected/dense/neural network layers) is excluded using ‘include_top = False’. This is because the pre-trained model is intended to be used as a ‘Feature Extractor’ and therefore, a bespoke classifier needs to be added on top of it. In this case, a new dense layer with softmax activation function needs to be added to the model, which will have a number of neurons equal to the number of classes in the current classification task.

 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/d603ce26-adb9-4ccc-8a04-255f1404358d)

Figure 5: ‘include_top = False’ depiction
o	Additionally, if ‘include_top=False’, the ‘input_shape’ parameter must also be specified when creating the model, which should match the shape of the new input data. 
o	the pooling parameter is set to 'avg', which means that the global average pooling operation is applied to the output of the last convolutional layer of the ResNet50 model. 
o	the weights parameter is set to 'imagenet', which means that the pre-trained weights on the ImageNet dataset are loaded into the model (for convolution layer only).
•	The trainability of the first layer (ResNet) of the model to be non-trainable (.trainable = False), which means the weights of the ResNet layer are frozen and won't be updated during training. This is common in transfer learning to preserve pre-trained knowledge.
•	Input and hidden Dense layers with ‘ReLU’ and output Dense layer with a ‘softmax’ activation function is added to my_model after the ResNet50 base model. This layer is used to make predictions and classify data into the specified number of classes (4 in this case).
•	my_model is then compiled using ‘Adam’ optimizer, ‘SparseCategoricalCrossentropy()’ loss function (4 different classes, not requiring one-hot encoding)
•	Finally, a summary of my_model, which provides information about the layers, output shapes, and the total number of parameters in the model is printed, as shown in Figure 6. The trainability of the ResNet layer is set to false, as can be seen in the summary.
 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/f0a61f34-0b68-44c8-912e-2086768a742f)

Figure 6: 'my_model' Summary
1.5.2.	Data Preparation:
The image data for the purpose of this report has been downloaded from Kaggle (Brain Tumor MRI classification V2 | Kaggle). The dataset contains a total of 7023 images, with training split of 5712 and testing split of 1311 images. The image classified into 4 classes. A snippet is shown in Figure 7.


 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/1d3f4fea-ff8b-477f-a0e7-8463a090ce16)

Figure 7: Input Data from Kaggle
As the ResNet50 (tf.keras.applications.resnet50.ResNet50) expects the input data to be pre-processed using tf.keras.applications.resnet50.preprocess_input. 
As shown in Figure 8:
•	two ImageDataGenerator objects using TensorFlow’s Keras API are created. The first object is created with the preprocessing_function parameter set to tf.keras.applications.resnet50.preprocess_input, which is a function that preprocesses a tensor or Numpy array encoding a batch of images. This function scales the pixel values of the input image to be between -1 and 1, which is required by the ResNet50 model.
•	The train_generator object is created using the flow_from_directory() method, which generates batches of tensor image data with real-time data augmentation. The directory parameter specifies the path to the training data directory, and the target_size parameter specifies the size of the input images. The batch_size parameter specifies the number of samples in each batch, and the class_mode parameter is set to 'sparse', which means that the labels are encoded as integers. 
•	The validation_generator object is created in a similar way, but with different parameters. It generates batches of validation data from a separate directory specified by the directory parameter.

The color_mode parameter is set to 'rgb', which means that the input images are expected to have 3 color channels. Even though, the input images are greyscale images, they are passed as rgb due to the fact that Resnet50 was trained with RGB images and expects 4D inputs(num_images, pixel_h, pixel_w, 3 colour channels).

•	The class_indices attribute of each generator object returns a dictionary mapping class names to class indices, this is done to ensure that the labels are correctly perceived by the model.

 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/c37d5283-326d-4aaf-bf0f-b4d9f058d052)

Figure 8: Image preprocessing and pipeline building
The processed and batched inputs are visualized using code, both shown in Figure 9.

 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/269d84a3-6dc1-4713-ab1d-f30a39855376)

Figure 9: Visualization of Inputs
1.5.3.	Training and Validation
The next step is to train the model on new data. The code (see Figure 10) trains a Keras model using the fit() method. 
•	The train_generator and validation_generator objects are used as input data for the model. 
•	The steps_per_epoch parameter is set to STEP_SIZE_TRAIN, which is the number of batches in the training set. 
•	The validation_steps parameter is set to STEP_SIZE_VALID, which is the number of batches in the validation set. 
•	The epochs parameter is set to 10, which means that the model will be trained for 10 epochs.
The fit() method trains the model for a fixed number of epochs (iterations on a dataset). For each epoch, it iterates over the training data in batches, computes the loss and gradients, and updates the model weights. After each epoch, it evaluates the model on the validation data, as shown in Figure 10.
The hist variable stores a history object that contains information about the training process, such as the loss and accuracy at each epoch (see Figure 11).




![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/91603afc-85fd-4c45-90e5-5de51ef5dca9)

Figure 10:Train, Validate and Save Model



 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/3410b328-c60c-4e96-a473-aa1d73ad2fdb)

Figure 11: Model undergoing training

1.5.4.	Results

The model achieved ~97 % val_accuracy as shown in Figure 12. 

 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/1d97ebc1-d041-401d-bd4d-64e3f813ab5c)

Figure 12: Validation Accuracy

The reduction in val_loss is shown in Figure 13.
 ![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/cefc7a08-e39d-4f54-a311-0067c9798821)
Figure 13: Validation Loss
