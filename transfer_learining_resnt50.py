# see report for explanation of the code
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)  # 2.14.0
num_classes = 4
my_model = tf.keras.Sequential()
my_model.add(tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(256, 256, 3), pooling='avg',
                                                     weights='imagenet'))
my_model.add(tf.keras.layers.Flatten())
my_model.add(tf.keras.layers.Dense(units=512, activation='relu'))
my_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
my_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
my_model.layers[0].trainable = False
for layer in my_model.layers:
    print(layer.trainable)
opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
my_model.compile(optimizer=opt_adam, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(my_model.summary())


# Prepare and load the data
datadir_train = "Training\\"
datadir_val = "Testing\\"

preprocess_input = tf.keras.applications.resnet50.preprocess_input
img_generator = tf.keras.preprocessing.image.ImageDataGenerator
data_generator = img_generator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    directory=datadir_train,
    target_size=(256, 256),
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb')
print(train_generator.class_indices)

validation_generator = data_generator.flow_from_directory(
    directory=datadir_val,
    target_size=(256, 256),
    batch_size=3,
    class_mode='sparse',
    color_mode='rgb')
print(validation_generator.class_indices)

x = train_generator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for i in range(0,4):
    image = x[0][i]
    ax[i].imshow(image)
plt.show()
plt.close()

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

hist = my_model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=10)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

my_model.save('my_model.keras')
my_model.save("E:\\1. All_Notes\\4. Thesis\\Masters\\transfer learning\\my_model\\brain_tumor_renet50.h5")
