# the following code categorizes a voice to one of angry, happy, sad, neutral, fear, disgust, surprise
# using the RAVDESS dataset

# ---- import libraries ---- #

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore') # ignore warnings that may appear during training

# open folder containing the dataset
folder_ = "/content/drive/MyDrive/Colab Notebooks/data"

dataset = os.listdir(folder_)

emotions = []
path = []

for file_ in dataset:
    try:
      actor = os.listdir(folder_ + '/' + file_)
      for voice in actor:
          x = voice.split('.')[0].split('-')
          emotions.append(int(x[2]))
          path.append(folder_ + '/' + file_ + '/' + voice)
    except:
      pass

assert len(emotions) == len(path), "Error occured!"

# create a dataframe with the emotions and path of each voice
emotion_data = pd.DataFrame(emotions, columns = ['Emotions'])
path_data = pd.DataFrame(path, columns = ['Path'])

new_data = pd.concat([emotion_data, path_data], axis = 1)

# create a dictionary to map emotions to their corresponding labels
emotion_dict = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
new_data.Emotions.replace(emotion_dict, inplace = True)
new_data.head()


S = set()
for x in new_data.Emotions:
    S.add(x)
print(S)


# ---- view data ---- #

new_data.Emotions.value_counts()
plt.title('Number of voices for each emotion')
plt.xlabel('Emotions')
plt.ylabel('Number of voices')
new_data.Emotions.value_counts().plot(kind = 'bar')
plt.show()



# ---- extract features from the voices ---- #

# in this part we extract the features from the voices using librosa library
def extract(data):
    sample_rate = 44100
    result = np.array([])
    result=np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0))) # stacking horizontally

    return result

def augment(path):

    def noise(data):
        noise_amp = 0.04*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(data, rate=0.8):
        return librosa.effects.time_stretch(data, rate = rate)

    def pitch(data, sample_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sr = sample_rate, n_steps = pitch_factor)

    def higher_speed(data, speed_factor=1.25):
        return librosa.effects.time_stretch(data, rate = speed_factor)

    def lower_speed(data, speed_factor=0.75):
        return librosa.effects.time_stretch(data, rate = speed_factor)

    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = np.array(extract(data))
    res = np.vstack((res, extract(noise(data))))
    # res = np.vstack((res, extract(stretch(data))))
    # res = np.vstack((res, extract(pitch(data, 44100))))

    return res

def transform_voice_to_image(path, cnt, emotion):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    if not os.path.exists('new_data'):
        os.makedirs('new_data')
    
    if not os.path.exists('new_data/'+emotion):
        os.makedirs('new_data/'+emotion)
    
    filename = "img"+str(cnt)
    plt.savefig('new_data/'+emotion+'/'+filename+'.png', dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()

img = transform_voice_to_image('/content/drive/MyDrive/Colab Notebooks/data/Actor_01/03-01-01-01-01-01-01.wav', 1)
#check image shape


X, Y = [], []

for i in range(len(new_data)):
    for x in augment(new_data.Path[i]):
        X.append(x)
        Y.append(new_data.Emotions[i])
        
len(X), len(Y), new_data.Path.shape

X1, Y1 = X[:], Y[:]

len(X1), len(Y1)

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
print(Features.head())

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

# ---- one hot encode the labels ---- #
onehot = OneHotEncoder()
Y = onehot.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# ---- split the data into training and testing sets ---- #
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)

X_train.shape, Y_train.shape

# ---- scale the data ---- #
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# we need to reduce overfitting so we can use dropout layers and batch normalization

model=Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))


model.add(Flatten())
# try another activation function like sigmoid
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dropout(0.3))

model.add(Dense(units=8, activation='softmax'))

# ---- compile the model ---- #
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---- train the model ---- #
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, mode='auto')

history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_test, Y_test), callbacks=[reduce_lr, early_stop])

# ---- plot the accuracy and loss ---- #
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'])
plt.show()


# ---- evaluate the model ---- #
loss, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: ', accuracy)
print('Loss: ', loss)

# ---- make predictions ---- #
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

# ---- classification report ---- #
print(classification_report(np.argmax(Y_test, axis=1), predictions))

# ---- confusion matrix ---- #
cm = confusion_matrix(np.argmax(Y_test, axis=1), predictions)
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

# ---- try to predict a voice ---- #

def predict_voice(path):
    # Load the audio file
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extract features
    features = np.array(extract(data))
    
    # Standardize the features
    features = scaler.transform([features])
    
    # Expand dimensions for model input
    features = np.expand_dims(features, axis=2)
    
    # Make predictions
    prediction = model.predict(features)
    
    predicted_emotion = emotion_dict[np.argmax(prediction)]

    return predicted_emotion

predict_voice('/content/drive/MyDrive/Colab Notebooks/data/Actor_01/03-01-01-01-01-01-01.wav')