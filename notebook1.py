#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


dir_sp_train = './spiral/training'
dir_sp_test = './spiral/testing'
dir_wv_train = './wave/training'
dir_wv_test = './wave/testing'


Name=[]
for file in os.listdir(dir_sp_train):
    Name+=[file]
print(Name)
print(len(Name))


N=[]
for i in range(len(Name)):
    N+=[i]
    
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

def mapper(value):
    return reverse_mapping[value]


dataset_sp=[]
count=0
for file in os.listdir(dir_sp_train):
    path=os.path.join(dir_sp_train,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        dataset_sp.append([image,count])
    count=count+1
    
testset_sp=[]
count=0
for file in os.listdir(dir_sp_test):
    path=os.path.join(dir_sp_test,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        testset_sp.append([image,count])
    count=count+1    


dataset_wv=[]
count=0
for file in os.listdir(dir_wv_train):
    path=os.path.join(dir_wv_train,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        dataset_wv.append([image,count])
    count=count+1
    
testset_wv=[]
count=0
for file in os.listdir(dir_wv_test):
    path=os.path.join(dir_wv_test,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        testset_wv.append([image,count])
    count=count+1   


data_sp,labels_sp0=zip(*dataset_sp)
test_sp,tlabels_sp0=zip(*testset_sp)

data_wv,labels_wv0=zip(*dataset_wv)
test_wv,tlabels_wv0=zip(*testset_wv)


labels_sp1=to_categorical(labels_sp0)
data_sp=np.array(data_sp)
labels_sp=np.array(labels_sp1)

tlabels_sp1=to_categorical(tlabels_sp0)
test_sp=np.array(test_sp)
tlabels_sp=np.array(tlabels_sp1)


labels_wv1=to_categorical(labels_wv0)
data_wv=np.array(data_wv)
labels_wv=np.array(labels_wv1)

tlabels_wv1=to_categorical(tlabels_wv0)
test_wv=np.array(test_wv)
tlabels_wv=np.array(tlabels_wv1)


trainx_sp,testx_sp,trainy_sp,testy_sp=train_test_split(data_sp,labels_sp,test_size=0.2,random_state=44)
trainx_wv,testx_wv,trainy_wv,testy_wv=train_test_split(data_wv,labels_wv,test_size=0.2,random_state=44)


print(trainx_sp.shape)
print(testx_sp.shape)
print(trainy_sp.shape)
print(testy_sp.shape)


datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")


pretrained_model1 = tf.keras.applications.DenseNet201(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model1.trainable = False

pretrained_model2 = tf.keras.applications.MobileNetV2(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model2.trainable = False


pretrained_model3 = tf.keras.applications.ResNet50(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model3.trainable = False

pretrained_model4 = tf.keras.applications.VGG16(input_shape=(100,100,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model4.trainable = False


inputs1 = pretrained_model1.input
x1 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model1.output)
outputs1 = tf.keras.layers.Dense(2, activation='softmax')(x1)
model1 = tf.keras.Model(inputs=inputs1, outputs=outputs1)

inputs2 = pretrained_model2.input
x2 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model2.output)
outputs2 = tf.keras.layers.Dense(2, activation='softmax')(x2)
model2 = tf.keras.Model(inputs=inputs2, outputs=outputs2)

inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(2, activation='softmax')(x3)
model3 = tf.keras.Model(inputs=inputs3, outputs=outputs3)

inputs4 = pretrained_model4.input
x4 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model4.output)
outputs4 = tf.keras.layers.Dense(2, activation='softmax')(x4)
model4 = tf.keras.Model(inputs=inputs4, outputs=outputs4)


model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


print("Training Spiral Set")

print("Start - 1...!")
his1s=model1.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_sp,testy_sp),epochs=50)
print("End 1")

print("Start - 2...!")
his2s=model2.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_sp,testy_sp),epochs=50)
print("End 2")

print("Start - 3...!")
his3s=model3.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_sp,testy_sp),epochs=50)
print("End 3")

print("Start - 4...!")
his4s=model4.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_sp,testy_sp),epochs=50)
print("End 4")

print("--------------------------------")
print("Training Wave Set")

print("Start - 1...!")
his1w=model1.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(testx_wv,testy_wv),epochs=50)
print("End 1")

print("Start - 2...!")
his2w=model2.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(testx_wv,testy_wv),epochs=50)
print("End 2")

print("Start - 3...!")
his3w=model3.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(testx_wv,testy_wv),epochs=50)
print("End 3")

print("Start - 4...!")
his4w=model4.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(testx_wv,testy_wv),epochs=50)
print("End 4")


#spiral1
y_pred_sp=model1.predict(testx_sp)
pred_sp=np.argmax(y_pred_sp,axis=1)
ground_sp = np.argmax(testy_sp,axis=1)
print(classification_report(ground_sp,pred_sp))

#spiral2
y_pred_sp=model2.predict(testx_sp)
pred_sp=np.argmax(y_pred_sp,axis=1)
ground_sp = np.argmax(testy_sp,axis=1)
print(classification_report(ground_sp,pred_sp))

#spiral3
y_pred_sp=model3.predict(testx_sp)
pred_sp=np.argmax(y_pred_sp,axis=1)
ground_sp = np.argmax(testy_sp,axis=1)
print(classification_report(ground_sp,pred_sp))

#spiral4
y_pred_sp=model4.predict(testx_sp)
pred_sp=np.argmax(y_pred_sp,axis=1)
ground_sp = np.argmax(testy_sp,axis=1)
print(classification_report(ground_sp,pred_sp))


#wave
y_pred_wv=model1.predict(testx_wv)
pred_wv=np.argmax(y_pred_wv,axis=1)
ground_wv = np.argmax(testy_wv,axis=1)
print(classification_report(ground_wv,pred_wv))

#wave
y_pred_wv=model2.predict(testx_wv)
pred_wv=np.argmax(y_pred_wv,axis=1)
ground_wv = np.argmax(testy_wv,axis=1)
print(classification_report(ground_wv,pred_wv))

#wave
y_pred_wv=model3.predict(testx_wv)
pred_wv=np.argmax(y_pred_wv,axis=1)
ground_wv = np.argmax(testy_wv,axis=1)
print(classification_report(ground_wv,pred_wv))

#wave
y_pred_wv=model4.predict(testx_wv)
pred_wv=np.argmax(y_pred_wv,axis=1)
ground_wv = np.argmax(testy_wv,axis=1)
print(classification_report(ground_wv,pred_wv))


get_acc1s = his1s.history['accuracy']
value_acc1s = his1s.history['val_accuracy']
get_loss1s = his1s.history['loss']
validation_loss1s = his1s.history['val_loss']


plt.figure()
plt.suptitle('Traning Results for model 1', fontsize=14)
epochs1s = range(len(get_acc1s))
plt.plot(epochs1s, get_acc1s, 'r', label='Accuracy of Training data')
plt.plot(epochs1s, value_acc1s, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Spiral')
plt.legend(loc=0)
plt.grid()


plt.figure()
epochs1s = range(len(get_loss1s))
plt.plot(epochs1s, get_loss1s, 'r', label='Loss of Training data')
plt.plot(epochs1s, validation_loss1s, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Spiral')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc2s = his2s.history['accuracy']
value_acc2s = his2s.history['val_accuracy']
get_loss2s = his2s.history['loss']
validation_loss2s = his2s.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 2', fontsize=14)
epochs2s = range(len(get_acc2s))
plt.plot(epochs1s, get_acc2s, 'r', label='Accuracy of Training data')
plt.plot(epochs1s, value_acc2s, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Spiral')
plt.legend(loc=0)
plt.grid()

plt.figure()
epochs2s = range(len(get_loss2s))
plt.plot(epochs2s, get_loss2s, 'r', label='Loss of Training data')
plt.plot(epochs2s, validation_loss2s, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Spiral')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc3s = his3s.history['accuracy']
value_acc3s = his3s.history['val_accuracy']
get_loss3s = his3s.history['loss']
validation_loss3s = his3s.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 3', fontsize=14)
epochs3s = range(len(get_acc3s))
plt.plot(epochs3s, get_acc3s, 'r', label='Accuracy of Training data')
plt.plot(epochs3s, value_acc3s, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Spiral')
plt.legend(loc=0)
plt.grid()


plt.figure()
epochs3s = range(len(get_loss3s))
plt.plot(epochs3s, get_loss3s, 'r', label='Loss of Training data')
plt.plot(epochs3s, validation_loss3s, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Spiral')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc4s = his4s.history['accuracy']
value_acc4s = his4s.history['val_accuracy']
get_loss4s = his4s.history['loss']
validation_loss4s = his4s.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 4', fontsize=14)
epochs4s = range(len(get_acc4s))
plt.plot(epochs4s, get_acc4s, 'r', label='Accuracy of Training data')
plt.plot(epochs4s, value_acc4s, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Spiral')
plt.legend(loc=0)
plt.grid()

plt.figure()
epochs4s = range(len(get_loss4s))
plt.plot(epochs4s, get_loss4s, 'r', label='Loss of Training data')
plt.plot(epochs4s, validation_loss4s, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Spiral')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc1w = his1w.history['accuracy']
value_acc1w = his1w.history['val_accuracy']
get_loss1w = his1w.history['loss']
validation_loss1w = his1w.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 1', fontsize=14)
epochs1w = range(len(get_acc1w))
plt.plot(epochs1w, get_acc1w, 'r', label='Accuracy of Training data')
plt.plot(epochs1w, value_acc1w, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Wave')
plt.legend(loc=0)
plt.grid()

plt.figure()
epochs1w = range(len(get_loss1w))
plt.plot(epochs1w, get_loss1w, 'r', label='Loss of Training data')
plt.plot(epochs1w, validation_loss1w, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Wave')
plt.legend(loc=0)
plt.grid()

plt.show()


get_acc2w = his2w.history['accuracy']
value_acc2w = his2w.history['val_accuracy']
get_loss2w = his2w.history['loss']
validation_loss2w = his2w.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 2', fontsize=14)
epochs2w = range(len(get_acc2w))
plt.plot(epochs2w, get_acc2w, 'r', label='Accuracy of Training data')
plt.plot(epochs2w, value_acc2w, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Wave')
plt.legend(loc=0)
plt.grid()

plt.figure()
epochs2w = range(len(get_loss2w))
plt.plot(epochs2w, get_loss2w, 'r', label='Loss of Training data')
plt.plot(epochs2w, validation_loss2w, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Wave')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc3w = his3w.history['accuracy']
value_acc3w = his3w.history['val_accuracy']
get_loss3w = his3w.history['loss']
validation_loss3w = his3w.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 3', fontsize=14)
epochs3w = range(len(get_acc3w))
plt.plot(epochs3w, get_acc3w, 'r', label='Accuracy of Training data')
plt.plot(epochs3w, value_acc3w, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Wave')
plt.legend(loc=0)
plt.grid()


plt.figure()
epochs3w = range(len(get_loss3w))
plt.plot(epochs3w, get_loss3w, 'r', label='Loss of Training data')
plt.plot(epochs3w, validation_loss3w, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Wave')
plt.legend(loc=0)
plt.grid()


plt.show()


get_acc4w = his4w.history['accuracy']
value_acc4w = his4w.history['val_accuracy']
get_loss4w = his4w.history['loss']
validation_loss4w = his4w.history['val_loss']

plt.figure()
plt.suptitle('Traning Results for model 4', fontsize=14)
epochs4w = range(len(get_acc4w))
plt.plot(epochs4w, get_acc4w, 'r', label='Accuracy of Training data')
plt.plot(epochs4w, value_acc4w, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy - Wave')
plt.legend(loc=0)
plt.grid()

plt.figure()
epochs4w = range(len(get_loss4w))
plt.plot(epochs4w, get_loss4w, 'r', label='Loss of Training data')
plt.plot(epochs4w, validation_loss4w, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss - Wave')
plt.legend(loc=0)
plt.grid()


plt.show()





load_img("./spiral/testing/parkinson/V03PE07.png",target_size=(512,512))


image=load_img("./spiral/testing/parkinson/V03PE07.png",target_size=(100,100))

image=img_to_array(image) 
image=image/255.0
prediction_image=np.array(image)
prediction_image=np.expand_dims(image, axis=0)

prediction=model1.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print(f"Prediction using model 1 is {move_name}.")

prediction=model2.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print(f"Prediction using model 2 is {move_name}.")

prediction=model3.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print(f"Prediction using model 3 is {move_name}.")

prediction=model4.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print(f"Prediction using model 4 is {move_name}.")


load_img("./wave/testing/parkinson/V03PO01.png",target_size=(512,512))


image2=load_img("./wave/testing/parkinson/V03PO01.png",target_size=(100,100))

image2=img_to_array(image2) 
image2=image2/255.0
prediction_image2=np.array(image2)
prediction_image2=np.expand_dims(image2, axis=0)

prediction2=model1.predict(prediction_image2)
value2=np.argmax(prediction2)
move_name2=mapper(value2)
print(f"Prediction using model 1 is {move_name2}.")

prediction2=model2.predict(prediction_image2)
value2=np.argmax(prediction2)
move_name2=mapper(value2)
print(f"Prediction using model 2 is {move_name2}.")

prediction2=model3.predict(prediction_image2)
value2=np.argmax(prediction2)
move_name2=mapper(value2)
print(f"Prediction using model 3 is {move_name2}.")

prediction2=model4.predict(prediction_image2)
value2=np.argmax(prediction2)
move_name2=mapper(value2)
print(f"Prediction using model 4 is {move_name2}.")


load_img("./spiral/testing/healthy/V01HE01.png",target_size=(512,512))


image3=load_img("./spiral/testing/healthy/V01HE01.png",target_size=(100,100))

image3=img_to_array(image3) 
image3=image3/255.0
prediction_image3=np.array(image3)
prediction_image3=np.expand_dims(image3, axis=0)

prediction3=model1.predict(prediction_image3)
value3=np.argmax(prediction3)
move_name3=mapper(value3)
print(f"Prediction using model 1 is {move_name3}.")

prediction3=model2.predict(prediction_image3)
value3=np.argmax(prediction3)
move_name3=mapper(value3)
print(f"Prediction using model 2 is {move_name3}.")

prediction3=model3.predict(prediction_image3)
value3=np.argmax(prediction3)
move_name3=mapper(value3)
print(f"Prediction using model 3 is {move_name3}.")

prediction3=model4.predict(prediction_image3)
value3=np.argmax(prediction3)
move_name3=mapper(value3)
print(f"Prediction using model 4 is {move_name3}.")


load_img("./wave/testing/healthy/V01HO01.png",target_size=(512,512))


image4=load_img("./wave/testing/healthy/V01HO01.png",target_size=(100,100))

image4=img_to_array(image4) 
image4=image4/255.0
prediction_image4=np.array(image4)
prediction_image4=np.expand_dims(image4, axis=0)

prediction4=model1.predict(prediction_image4)
value4=np.argmax(prediction4)
move_name4=mapper(value4)
print(f"Prediction using model 1 is {move_name4}.")

prediction4=model2.predict(prediction_image4)
value4=np.argmax(prediction4)
move_name4=mapper(value4)
print(f"Prediction using model 2 is {move_name4}.")

prediction4=model3.predict(prediction_image4)
value4=np.argmax(prediction4)
move_name4=mapper(value4)
print(f"Prediction using model 3 is {move_name4}.")

prediction4=model4.predict(prediction_image4)
value4=np.argmax(prediction4)
move_name4=mapper(value4)
print(f"Prediction using model 4 is {move_name4}.")





import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# define the directory path and image dimensions
root_test_dir = './drawings/'
img_width, img_height = 100, 100

# define a function to get the predicted label from the model's output
def get_label(prediction):
    if prediction[0] < prediction[1]:
        return 'parkinson'
    else:
        return 'healthy'


# initialize variables to keep track of the number of correctly predicted labels
models = [model1, model2, model3, model4]
num_correct = 0
total_images = 0
prevs=[0]

for i, model in enumerate(models):
    j=0
    print(f'Predictions using model {i+1}:')
    # iterate through all the images in the testing directory
    for subdir1, dirs1, files1 in os.walk(root_test_dir):
        for subdir2, dirs2, files2 in os.walk(subdir1):
            if j==4:
                break
            if "testing\\" in subdir2:
                j+=1
                for file in files2:
                    # load the image
                    image = load_img(os.path.join(subdir2, file), target_size=(img_width, img_height))
                    
                    image=img_to_array(image) 
                    
                    image=image/255.0
                    
                    prediction_image=np.array(image)
                    
                    prediction_image=np.expand_dims(image, axis=0)
                    
                    prediction=model.predict(prediction_image,verbose=0)
                    
                    value=np.argmax(prediction)
                    
                    predicted_label=mapper(value)
                    # get the actual label from the directory structure
                    actual_label = os.path.basename(subdir2)
                    
                    # increment the counters based on whether the prediction was correct
                    if predicted_label == actual_label:
                        num_correct += 1
                    total_images += 1
    # print the accuracy for this model
    
    accuracy = num_correct / total_images
    print(f'Accuracy using model {i+1} is {accuracy:.2f} (scored {num_correct}/{total_images})')
    prevs.append(num_correct)
    num_correct = 0
    total_images = 0


# saving models so next time we can load them directly instead of training again
model1.save("model1.h5")
model2.save("model2.h5")
model3.save("model3.h5")
model4.save("model4.h5")

# loading models
#model1 = load_model("model1.h5")
#model2 = load_model("model2.h5")
#model3 = load_model("model3.h5")
#model4 = load_model("model4.h5")







