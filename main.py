import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"]="tf.keras"
from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import keras
class DataGetter:
    def __init__(self, path: str):
        self.dir_path = path
        ids = os.listdir(path)
        self.images = [os.path.join(path, image_id) for image_id in ids]
        self.data = []
    def get_data(self):
        for i in range(len(self.images)):
            im = cv2.imread(self.images[i])
            conv_im = cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.data.append(conv_im)
        return np.array(self.data)

train_data = DataGetter("./cropped_data/train_data").get_data()
train_masks = DataGetter("./cropped_data/train_mask").get_data()
val_data = DataGetter("./cropped_data/val_data").get_data()
val_masks = DataGetter("./cropped_data/val_mask").get_data()
#test_data = DataGetter("./cropped_data/test_data").get_data()
#test_masks = DataGetter("./cropped_data/test_mask").get_data()
print(train_masks.shape)
BACKBONE = 'resnet18'
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE)


# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile("Adam", total_loss, metrics)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]


history = model.fit(
    x = train_data,
    y = train_masks, 
    epochs=EPOCHS, 
    callbacks=callbacks,
    validation_data = (val_data, val_masks)
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