#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras import layers,models


# In[ ]:





# In[5]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.cifar100.load_data()


# In[3]:


X_train.shape


# In[4]:


plt.matshow(X_train[0])


# In[5]:


y_train


# In[6]:


y_train.shape


# In[8]:


y_train.reshape(-1,)
y_test.reshape(-1,)


# In[6]:


X_train=X_train/255.0


# In[7]:


X_test=X_test/255.0


# In[10]:


y_train


# In[11]:


y_train.shape


# In[12]:


y_train.reshape(-1,)


# In[13]:


y_train.shape


# In[14]:


y_train


# In[9]:


y_train=y_train.reshape(-1,)
y_test=y_test.reshape(-1,)


# In[16]:


y_train[np.argmax(y_train)]


# In[17]:


superclasses=["aquatic mammals",
"fish",
"flowers",
"food containers",
"fruit and vegetables",
"household electrical devices",
"household furniture",
"insects",
"large carnivores",
"large man-made outdoor things",
"large natural outdoor scenes",
"large omnivores and herbivores",
"medium-sized mammals",
"non-insect invertebrates",
"people",
"reptiles",
"small mammals",
"trees",
"vehicles 1",
"vehicles 2",
]


# In[18]:


superclasses[0]


# In[19]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3),padding="Same", activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
   
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same",activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
     
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(100, activation='softmax')
])


# In[20]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=1)


# In[21]:


X_train


# In[22]:


cnn.evaluate(X_test,y_test)


# In[23]:


predictions=cnn.predict(X_train)
predictions


# In[24]:


np.argmax(predictions[0])


# In[25]:


y_train


# In[ ]:




