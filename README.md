# Image Iterator fir Keras Image Preprocessing
Loads batches of generated images with one-hot encoded labels. Intended to be used with keras `ImageDataGenerator`. 
Useful to iterate over the list of files with corresponding labels.

## Usage
```Python
from keras_img_iterator import SingleDirectoryIterator

gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = SingleDirectoryIterator(
    directory='../data/train_img/',
    filenames=data['file'],
    labels=data['label'],
    image_data_generator=gen,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    seed=1337)
```
## Training

```Python
model.fit_generator(
    train_generator,
    steps_per_epoch=num_samples // batch_size,
    epochs=50)
```
