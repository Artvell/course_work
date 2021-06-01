import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"]="tf.keras"
from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import keras
"""class DataGetter:
    def __init__(self, path: str):
        self.dir_path = path
        ids = os.listdir(path)
        self.images = [os.path.join(path, image_id) for image_id in ids]
        self.data = []
    def get_data(self):
        for i in range(len(self.images)):
            im = cv2.imread(self.images[i])
            conv_im = im.astype(np.int16)
            self.data.append(conv_im)
        print("uint",im)
        print("int8",conv_im)
        plt.imshow(conv_im, cmap = 'Blues', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        return self.data

train_data = DataGetter("./cropped_data/train_data").get_data()
train_masks = DataGetter("./cropped_data/train_mask").get_data()
val_data = DataGetter("./cropped_data/val_data").get_data()
val_masks = DataGetter("./cropped_data/val_mask").get_data()
#test_data = DataGetter("./cropped_data/test_data").get_data()
#test_masks = DataGetter("./cropped_data/test_mask").get_data()
#print(train_masks.shape)





# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
"""
DATA_DIR = './cropped_data/'

x_train_dir = os.path.join(DATA_DIR, 'train_data')
y_train_dir = os.path.join(DATA_DIR, 'train_mask')

x_valid_dir = os.path.join(DATA_DIR, 'val_data')
y_valid_dir = os.path.join(DATA_DIR, 'val_mask')

x_test_dir = os.path.join(DATA_DIR, 'test_data')
y_test_dir = os.path.join(DATA_DIR, 'test_mask')

def visualize(**images):
    """PLot images in one row."""
    """n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()"""
    cv2.imshow("!!!",images[0])
    cv2.waitKey(0)
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    

# classes for data loading and preprocessing
class Dataset:
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
    ):
        self.ids = os.listdir(images_dir)
        self.masks_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.masks_ids]
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float16)
        mask = mask.astype(np.float16)
        print("types: ", image.dtype, mask.dtype)
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

dataset = Dataset(x_train_dir, y_train_dir)

image, mask = dataset[5] # get some sample
visualize(
    image=image, 
    cars_mask=mask
)

BACKBONE = 'resnet18'
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 40

total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE,input_shape = (512, 512, 3))
# compile keras model with defined optimozer, loss and metrics
model.compile("Adam", total_loss, metrics)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min')
    #keras.callbacks.ReduceLROnPlateau(),
]

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS, 
    callbacks=callbacks,
    validation_data = valid_dataloader,
    validation_steps=len(valid_dataloader),
)

plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()