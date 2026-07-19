import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class ResNetClassifier:
    """
    Builds, trains, evaluates, and saves a ResNet50-based transfer learning model
    for image classification (specifically brain tumor scans).
    """
    def __init__(self, train_dir, val_dir, model_dir, log_dir, checkpoint_dir, 
                 batch_size=32, epochs=25, learning_rate=0.0001, patience=4, weights='imagenet'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.weights = weights

        self.class_indices = {}
        self.class_names = []
        self.train_generator = None
        self.validation_generator = None
        self.model = None

        self._configure_gpu()

    def _configure_gpu(self):
        """Enable memory growth for GPUs if available."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available and configured: {gpus}")
        except Exception as e:
            print(f"Error configuring GPU memory growth: {e}")

    def prepare_data(self, visualize_samples=True):
        """Prepares the training and validation data streams."""
        print("Preparing data generators...")
        
        # Initialize generators with preprocessing
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = data_generator.flow_from_directory(
            directory=self.train_dir,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='sparse',
            color_mode='rgb'
        )

        self.validation_generator = data_generator.flow_from_directory(
            directory=self.val_dir,
            target_size=(256, 256),
            batch_size=min(3, self.batch_size), # Or match the batch size if testing
            class_mode='sparse',
            color_mode='rgb'
        )

        self.class_indices = self.train_generator.class_indices
        self.class_names = sorted(list(self.class_indices.keys()))
        print(f"Detected Classes and Indices: {self.class_indices}")

        # Save some visual samples to check data loading without blocking execution
        if visualize_samples and self.model_dir:
            try:
                os.makedirs(self.model_dir, exist_ok=True)
                x = self.train_generator.next()
                fig, ax = plt.subplots(ncols=min(4, len(x[0])), figsize=(20, 5))
                # If only 1 image in batch
                if min(4, len(x[0])) == 1:
                    ax = [ax]
                for i in range(min(4, len(x[0]))):
                    image = x[0][i]
                    # Rescale image to [0, 1] range for visualization since resnet50 preprocess shifts range
                    image_min = image.min()
                    image_max = image.max()
                    if image_max > image_min:
                        image = (image - image_min) / (image_max - image_min)
                    ax[i].imshow(image)
                    ax[i].set_title(f"Label Index: {int(x[1][i])}")
                sample_path = os.path.join(self.model_dir, 'dataset_samples.png')
                plt.savefig(sample_path)
                plt.close()
                print(f"Saved dataset visualization samples to: {sample_path}")
            except Exception as e:
                print(f"Failed to visualize data samples: {e}")

    def build_model(self):
        """Constructs the ResNet50 Transfer Learning Sequential model."""
        num_classes = len(self.class_names) if self.class_names else 4
        print(f"Building model with ResNet50 backbone (weights={self.weights}) for {num_classes} classes...")

        self.model = Sequential()
        
        # Load pre-trained ResNet50
        resnet_backbone = ResNet50(
            include_top=False, 
            input_shape=(256, 256, 3), 
            pooling='avg',
            weights=self.weights
        )
        resnet_backbone.trainable = False # Freeze backbone weights
        
        self.model.add(resnet_backbone)
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=num_classes, activation='softmax'))

        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=opt, 
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
        print(self.model.summary())

    def train(self, save_plots=True):
        """Runs the model training pipeline."""
        if self.model is None:
            self.build_model()

        if self.train_generator is None or self.validation_generator is None:
            raise ValueError("Data generators are not initialized. Call prepare_data() first.")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Callbacks
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_dir, 
            monitor='val_accuracy', 
            save_best_only=True
        )

        callbacks = [tensorboard_cb, early_stop_cb, checkpoint_cb]

        step_size_train = self.train_generator.n // self.train_generator.batch_size
        step_size_valid = self.validation_generator.n // self.validation_generator.batch_size

        # In case generators have fewer images than batch size
        if step_size_train == 0:
            step_size_train = 1
        if step_size_valid == 0:
            step_size_valid = 1

        print("Starting training...")
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=step_size_train,
            validation_data=self.validation_generator,
            validation_steps=step_size_valid,
            epochs=self.epochs,
            callbacks=callbacks
        )
        print("Training completed successfully.")

        if save_plots:
            self._save_training_plots(history)

        return history

    def _save_training_plots(self, history):
        """Generates and saves performance curve plots."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Loss curves
        fig1 = plt.figure()
        plt.plot(history.history['loss'], color='teal', label='loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], color='orange', label='val_loss')
        fig1.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.model_dir, 'resnet_loss_curves.png'))
        plt.close(fig1)

        # Accuracy curves
        fig2 = plt.figure()
        plt.plot(history.history['accuracy'], color='teal', label='accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
        fig2.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.model_dir, 'resnet_accuracy_curves.png'))
        plt.close(fig2)
        print(f"Saved performance plots to {self.model_dir}")

    def evaluate(self):
        """Evaluates the model on validation data and prints the metrics."""
        if self.model is None or self.validation_generator is None:
            print("Model or validation data not loaded.")
            return None

        scores = self.model.evaluate(self.validation_generator, verbose=0)
        print(f"Validation loss: {scores[0]:.4f}")
        print(f"Validation accuracy: {scores[1]:.4f}")
        return scores

    def save_model(self, model_name='brain_tumor_resnet50.keras'):
        """Saves the final trained model."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_save_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_save_path)
        print(f"Model successfully saved to {model_save_path}")
        return model_save_path
