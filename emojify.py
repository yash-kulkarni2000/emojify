import numpy as np
import tensorflow as tf
import keras
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm


import cv2
import face_recognition
from PIL import Image 



train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'train/',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test/',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(3.), activation='relu', input_shape=(48,48,1)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(32, kernel_constraint=max_norm(3.), kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(32, kernel_constraint=max_norm(3.), kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(512, kernel_constraint=max_norm(3.), activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Dense(7, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)

cnn.summary()



# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    
    
    
    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        data = Image.fromarray(frame).convert('LA')
        
        for top, right, bottom, left in face_locations: 
            box = (left, top, right, bottom)
        
        cropped_image = data.crop(box)
        final_image = cropped_image.resize((48, 48))
        break
            
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()





final_image.save("def.png")

prediction_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

prediction_image = train_datagen.flow_from_directory(
        'pred/',
        target_size=(48,48),
        color_mode='grayscale',
        classes=None)


emotion_prediction = cnn.predict(prediction_image)
final_prediction = int(np.argmax(emotion_prediction))



emotion_list = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

print('The final prediction is ', emotion_list[final_prediction])

'''
numpydata = tf.keras.preprocessing.image.img_to_array(final_image, data_format='channels_last')

numpydata_1 = numpydata[:, :, 0]

numpydata_1.reshape([-1, 48, 48])
numpydata_1.shape

'''


