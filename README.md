# ResNet50 Transfer Learning Brain Tumor Classifier

A clean, modular, and fully Object-Oriented pipeline for brain tumor image classification using ResNet50 transfer learning. The system integrates custom Dense classifier layers on top of a frozen pre-trained ResNet50 backbone, packaged into a unified in-process orchestrator.

---

## 🚀 Key Features

* **Object-Oriented Design (OOP)**: Replaced procedural scripts with modular, reusable classes for training, configurations, and inference.
* **Installable Package**: Can be installed editably via pip (`pip install -e .`) and imported in any Python script.
* **On-the-Fly Predictor**: Runs inference directly on individual MRI brain scan images, returning predicted class names and confidence probabilities.
* **Automatic Bootstrapping**: Setup demo scripts to automatically bootstrap missing dependencies, install the package, and run a quick verification test.
* **Interactive Fallbacks**: Prompts for training, testing, and output folders graphically using `easygui` dialogue boxes if paths are not configured in `.ini` files.
* **Comprehensive Test Suite**: Integrates unittest coverage to test config loading, model building, dataset pipelines, and predictors entirely offline.

---

## 🧠 Transfer Learning with ResNet50

ResNet50 knowledge can be transferred to a target job using multiple techniques. The pre-trained model is utilized to begin the new task, and the model's weights are updated to fit the new dataset. 

For the purpose of this pipeline, the pre-trained ResNet50 model is used as a **Feature Extractor** that feeds the output to a bespoke classifier (i.e., fully connected layers). The steps involved in this are discussed below:

### 1. Pre-trained ResNet50 Model Selection & Architecture

Most of the pretrained models are readily available in deep learning frameworks such as TensorFlow/Keras or PyTorch. For the purposes of this pipeline, we have selected the ResNet50 model, which has been pre-trained on a source task (i.e., the ImageNet database) using Keras, as depicted in Figure 4.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/3d6636fe-d1e1-49ad-9220-bd7887e44523)

*Figure 4: Pre-Selection of ResNet50 model*

* The code uses TensorFlow (`tf`) to create a neural network model.
* The number of classes is set to 4, as this model is designed for a classification task with four brain tumor classes.
* A new sequential model is created using `tf.keras.Sequential()`.
* The ResNet50 model from Keras's applications module is added as the base.
* The top layer (fully connected/dense layers) is excluded using `include_top=False`. This is because the pre-trained model is intended to be used as a "Feature Extractor", and therefore, a bespoke classifier needs to be added on top of it.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/d603ce26-adb9-4ccc-8a04-255f1404358d)

*Figure 5: 'include_top = False' depiction*

* Additionally, when `include_top=False`, the `input_shape` parameter must also be specified when creating the model, which should match the shape of the new input data (configured to `(256, 256, 3)`).
* The `pooling` parameter is set to `'avg'`, which means that the global average pooling operation is applied to the output of the last convolutional layer of the ResNet50 model.
* The `weights` parameter is set to `'imagenet'`, which means that the pre-trained weights on the ImageNet dataset are loaded into the model (for convolutional layers only).
* The trainability of the first layer (ResNet backbone) of the model is set to non-trainable (`.trainable = False`), which means the weights of the ResNet layer are frozen and won't be updated during training. This is common in transfer learning to preserve pre-trained knowledge.
* Input and hidden Dense layers with `ReLU` activations and an output Dense layer with a `softmax` activation function are added to the model after the ResNet50 base to classify data into the specified number of classes (4 in this case).
* The model is compiled using the `Adam` optimizer (learning rate `0.0001`) and `SparseCategoricalCrossentropy()` loss function.
* Finally, a summary of the model, which provides information about the layers, output shapes, and the total number of parameters in the model is printed, as shown in Figure 6.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/f0a61f34-0b68-44c8-912e-2086768a742f)

*Figure 6: 'my_model' Summary*

---

## 📊 Dataset & Preparation

This pipeline is designed and tested using the **Brain Tumor MRI Classification** dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/masoudmzb/brain-tumor-mri-dataset). The dataset contains a total of 7,023 images, with a training split of 5,712 and a testing split of 1,311 images across 4 tumor classes: `glioma`, `meningioma`, `notumor`, and `pituitary`.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/1d3f4fea-ff8b-477f-a0e7-8463a090ce16)

*Figure 7: Input Data from Kaggle*

### Image Preprocessing and Generators:

As the ResNet50 model expects the input data to be pre-processed using the model-specific `preprocess_input` function:
* Two `ImageDataGenerator` objects using TensorFlow’s Keras API are created. The generator is initialized with the `preprocessing_function` parameter set to `tf.keras.applications.resnet50.preprocess_input`, which scales the pixel values of the input image to be between -1 and 1, as required by the ResNet50 model.
* The `train_generator` object generates batches of tensor image data with real-time data augmentation. The directory path to the training data is specified, along with the `target_size=(256, 256)` and `class_mode='sparse'`.
* The `validation_generator` is created in a similar way to generate batches of validation data.
* The `color_mode` parameter is set to `'rgb'`, which means that the input images are expected to have 3 color channels. Even though the input images are greyscale images, they are passed as RGB because ResNet50 expects 3-channel input arrays.
* The `class_indices` mapping is checked to ensure labels are aligned correctly.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/c37d5283-326d-4aaf-bf0f-b4d9f058d052)

*Figure 8: Image preprocessing and pipeline building*

The processed and batched inputs are visualized as shown in Figure 9.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/269d84a3-6dc1-4713-ab1d-f30a39855376)

*Figure 9: Visualization of Inputs*

---

## 📈 Training, Validation & Results

The training code triggers the Keras `fit()` method using the generators:
* The `train_generator` and `validation_generator` objects are used as input data streams.
* The `steps_per_epoch` is configured based on batch sizes.
* Training executes with `TensorBoard`, `EarlyStopping`, and `ModelCheckpoint` callbacks.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/91603afc-85fd-4c45-90e5-5de51ef5dca9)

*Figure 10: Train, Validate and Save Model*

The history object stores information about the training metrics, such as the loss and accuracy at each epoch, as shown in Figure 11.

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/3410b328-c60c-4e96-a473-aa1d73ad2fdb)

*Figure 11: Model undergoing training*

### Final Metrics & Performance:

The model achieves ~97% validation accuracy during a full training cycle:

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/1d97ebc1-d041-401d-bd4d-64e3f813ab5c)

*Figure 12: Validation Accuracy*

The corresponding reduction in validation loss over training epochs is depicted below:

![image](https://github.com/Vedaprakash88/ResNet50_Demo/assets/103208134/cefc7a08-e39d-4f54-a311-0067c9798821)

*Figure 13: Validation Loss*

---

## 🛠️ Installation

1. **Install Package Editably**:
   Activate your virtual environment and run the following command in the repository root directory:
   ```bash
   pip install -e .
   ```
   *(Dependencies include: `easygui`, `numpy`, `matplotlib`, `opencv-python`, `tensorflow`, `tqdm`)*

---

## ⚙️ Configuration Setup

To configure directories and training parameters, use the `.ini` configuration format:

1. **`config.ini`**: Provided in the repository with default hyperparameters. Fill in `train_dir`, `val_dir`, and `output_dir` before executing.
2. **`config.local.ini`** *(Recommended)*: Create a copy named `config.local.ini` at the root. The configuration loader prioritizes `config.local.ini` and it is already ignored in git to keep your local directory paths private.

### Example Configuration:
```ini
[paths]
train_dir = C:/path/to/Data_brain_tumor/Training
val_dir = C:/path/to/Data_brain_tumor/Testing
output_dir = C:/path/to/Data_brain_tumor/Outputs

[training]
batch_size = 32
epochs = 25
learning_rate = 0.0001
model_name = brain_tumor_resnet50.keras
patience = 4
```

---

## 📖 Usage Manual

### 1. Unified Pipeline (Run Training)
To process the datasets, compile the Sequential ResNet50 architecture, train the model, and save performance plots:

Run the main runner script at the root:
```bash
python main.py
```
*(If directories are not specified in the `.ini` files, easygui graphical prompt boxes will automatically open to guide folder selection).*

You can also run the orchestrator directly in any Python script:
```python
from resnet_classifier import load_config, ResNetOrchestrator

# Load configuration
config = load_config()

# Instantiate and run
orchestrator = ResNetOrchestrator(config)
orchestrator.run_pipeline(run_training=True)
```

---

### 2. Predictor Inference (Classify individual scans)
You can run inferences on new MRI images using the `ResNetPredictor` class. The predictor automatically reads, resizes, and pre-processes inputs.

```python
from resnet_classifier import ResNetPredictor

# Initialize predictor with your saved model path
predictor = ResNetPredictor(
    model_path="brain_tumor_resnet50.keras", 
    model_dir="C:/path/to/Data_brain_tumor/Outputs/my_model"
)

# Predict directly from an MRI image path
class_idx, class_name, probabilities = predictor.get_predicted_class(
    image_path="C:/path/to/new_scan.jpg",
    class_names=["glioma", "meningioma", "notumor", "pituitary"]
)

print(f"Predicted Tumor Type: {class_name} (Probabilities: {probabilities})")
```

---

### 3. Running Unit Tests
The repository includes a comprehensive unit test suite to verify configuration parsing, model construction, and in-memory prediction:

Run all tests via the test runner script:
```bash
python run_tests.py
```
Or directly using Python's unittest module:
```bash
python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```text
├── main.py                     # Root entry point script (with GUI dialog fallbacks)
├── config.ini                  # Template configuration file
├── config.local.ini            # Gitignored local configuration (optional)
├── pyproject.toml              # Library packaging configuration
├── requirements.txt            # System dependencies list
├── run_tests.py                # Unit test runner script
├── tests/                      # Package unit test suite
│   ├── __init__.py
│   ├── test_config.py          # Verifies config load settings
│   ├── test_classifier.py      # Verifies data streams and classifier pipeline
│   └── test_predictor.py       # Verifies predictor class loader and predictions
└── resnet_classifier/          # Core library package directory
    ├── __init__.py             # Exposed package imports
    ├── config.py               # Configuration file loader
    ├── classifier.py           # Preprocesses image datasets, builds/trains ResNet50 model
    ├── predictor.py            # Performs single-image classification inference
    └── orchestrator.py         # Drives the modular classifier steps
```

---

## 📋 Terminal Logs

```text
🎬 STARTING RESNET50 TRANSFER LEARNING DEMO & TEST
Reading configuration from: config.local.ini
Overriding epochs to 2 for a quick run.

🚀 RESNET50 TRANSFER LEARNING ORCHESTRATOR STARTED

Configured Training Directories:
  - Train Directory:      [dataset_root]/Data_brain_tumor/Training/
  - Validation Directory: [dataset_root]/Data_brain_tumor/Testing/
  - Model Save Directory: [dataset_root]/Data_brain_tumor\my_model
Preparing data generators...
Found 5712 images belonging to 4 classes.
Found 1311 images belonging to 4 classes.
Detected Classes and Indices: {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
Saved dataset visualization samples to: [dataset_root]/Data_brain_tumor\my_model\dataset_samples.png
Building model with ResNet50 backbone (weights=imagenet) for 4 classes...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 2048)              23587712  
                                                                 
 flatten (Flatten)           (None, 2048)              0         
                                                                 
 dense (Dense)               (None, 512)               1049088   
                                                                 
 dense_1 (Dense)             (None, 128)               65664     
                                                                 
 dense_2 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 24,702,980
Trainable params: 1,115,268
Non-trainable params: 23,587,712
_________________________________________________________________
None
Starting training...
Epoch 1/2
178/178 [==============================] - 47s 261ms/step - loss: 0.2186 - accuracy: 0.9229 - val_loss: 0.2920 - val_accuracy: 0.8955
Training completed successfully.
Saved performance plots to [dataset_root]/Data_brain_tumor\my_model
Validation loss: 0.2920
Validation accuracy: 0.8955
Model successfully saved to [dataset_root]/Data_brain_tumor\my_model\brain_tumor_resnet50.keras

===============================================================================
🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!
===============================================================================

--- TESTING ON-THE-FLY INFERENCE ---
Running inference on: [dataset_root]/Data_brain_tumor/Testing/glioma\Te-glTr_0000.jpg
Loading ResNet model from: [dataset_root]/Data_brain_tumor\my_model\brain_tumor_resnet50.keras
Model loaded successfully.

========================================
🔮 PREDICTION RESULT
========================================
Image Path: [dataset_root]/Data_brain_tumor/Testing/glioma\Te-glTr_0000.jpg
Predicted Class Index: 0
Predicted Class Name:  glioma
Prediction Probabilities: [0.9770742058753967, 0.021867720410227776, 0.0006272021564655006, 0.0004308246134314686]
===============================================================================
```
