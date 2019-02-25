from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Activation,Dropout,Input,Lambda,Cropping2D
import PIL
from PIL import Image
from keras.backend import tf as ktf


#Read in the drive log
df_drive_log = pd.read_csv('./driving_log.csv')



####appending the images for training
all_center_images = []
steering_angles = []
for index, row in df_drive_log.iterrows():
    ### READING CENTER IMAGES
    img = plt.imread(row['Centre'])
    all_center_images.append(img)
    all_center_images.append(np.fliplr(img))
    steering_angles.append(row['Steering'])
    steering_angles.append(-row['Steering'])

    correction = 0.2
    ### READING LEFT IMAGES
    img = plt.imread(row['Left'].lstrip())
    all_center_images.append(img)
    all_center_images.append(np.fliplr(img))
    steering_angles.append(row['Steering']+correction)
    steering_angles.append(-(row['Steering']+correction))
 
    ### READING LEFT IMAGES
    img = plt.imread(row['Right'].lstrip())
    all_center_images.append(img)
    all_center_images.append(np.fliplr(img))
    steering_angles.append(row['Steering']-correction)
    steering_angles.append(-(row['Steering']-correction))
    
all_center_images = np.asarray(all_center_images)    
steering_angles = np.asarray(steering_angles)



#####Prepare the train data
X_train = all_center_images
y_train = steering_angles


#Instantiate an empty model
model = Sequential()

##Preprocessing Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(110,320,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
#model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='valid'))
#model.add(Activation('relu'))

# 5th Convolutional Layer
#model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid'))
#model.add(Activation('relu'))
# Max Pooling
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
#model.add(Dense(4096))
#model.add(Activation('relu'))
# Add Dropout
#model.add(Dropout(0.4))

# 3rd Fully Connected Layer
#model.add(Dense(1000))
model.add(Dense(64))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
#model.add(Activation('softmax'))

model.summary()



# Compile the model
model.compile(loss='mse', optimizer='adam')


model.fit(X_train,y_train,epochs=5,shuffle=True,validation_split=0.1)
model.save('./drive_model.h5')