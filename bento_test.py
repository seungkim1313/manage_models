import bentoml
#import tensorflow as tf
#loaded_model = tf.keras.models.load_model("mobilenetv2_imagenet.h5")
#tag = bentoml.tensorflow.save('tf_mobilenetv2_imagenet', loaded_model)
import torch

loaded_model = torch.load("./vangogh_test/model.pth")
tag = bentoml.pytorch.save('cyclegan_vangogh', loaded_model)

metadata = bentoml.models.get(tag)
print(metadata)
