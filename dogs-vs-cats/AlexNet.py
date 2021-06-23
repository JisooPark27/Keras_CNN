#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
# 랜덤시드 고정시키기
np.random.seed(3)


# In[2]:


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('./dogs-vs-cats/train',
                                                    target_size=(227, 227),
                                                    batch_size=64,
                                                    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('./dogs-vs-cats/test',
                                                  target_size=(227, 227), 
                                                  batch_size=128,
                                                  class_mode='categorical')


# In[3]:



model = Sequential()
model.add(Conv2D(input_shape=(227,227,3),
                 filters = 96,
                 kernel_size=(11, 11),
                 padding='valid',
                 strides= 4,
                 activation='relu',
                 ))
model.add(MaxPooling2D(pool_size=(3, 3),
                      strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters = 256,
                 kernel_size = (5, 5),
                 padding='same',
                 strides=(1,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                      strides=(2,2)))
model.add(Conv2D(filters = 384,
                 kernel_size = (3, 3),
                 padding='same',
                 strides=(1,1),
                 activation='relu'))
model.add(Conv2D(filters = 384,
                 kernel_size = (3, 3),
                 padding='same',
                 strides=(1,1),
                 activation='relu'))
model.add(Conv2D(filters = 256,
                 kernel_size = (3, 3),
                 padding='same',
                 strides=(1,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                      strides=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))





# In[4]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


SVG(model_to_dot(model, show_shapes=True, dpi=70).create(prog='dot', format='svg'))


# In[5]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[6]:


model.fit_generator(train_generator,
                    steps_per_epoch=50,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=5)


# In[7]:


print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

