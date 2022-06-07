import tensorflow as tf

base_learning_rate = 0.0001

loaded_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
loaded_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
loaded_model.summary()

loaded_model.save('mobilenetv2_imagenet.h5')
