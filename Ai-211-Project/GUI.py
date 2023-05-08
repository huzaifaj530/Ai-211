import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
import librosa
from joblib import load
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# load the trained model
model = keras.models.load_model('my_model.h5')
le = LabelEncoder()
le.classes_ = np.load('classes.npy',allow_pickle=True)
fit=load('scaler.joblib')
# print(le.classes_)

# create a function to extract features from an audio file
def extract_features(filename):
    y, sr = librosa.load(filename,duration=30)
    length=66149
    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).var()
    y_harmonic = librosa.effects.harmonic(y)
    harmony_mean = np.mean(y_harmonic)
    harmony_var = np.var(y_harmonic)
    y_percussive = librosa.effects.percussive(y)
    perceptr_mean = np.mean(y_percussive)
    perceptr_var = np.var(y_percussive)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc1_mean = mfccs[0].mean()
    mfcc1_var = mfccs[0].var()
    mfcc2_mean = mfccs[1].mean()
    mfcc2_var = mfccs[1].var()
    mfcc3_mean = mfccs[2].mean()
    mfcc3_var = mfccs[2].var()
    mfcc4_mean = mfccs[3].mean()
    mfcc4_var = mfccs[3].var()
    mfcc5_mean = mfccs[4].mean()
    mfcc5_var = mfccs[4].var()
    mfcc6_mean = mfccs[5].mean()
    mfcc6_var = mfccs[5].var()
    mfcc7_mean = mfccs[6].mean()
    mfcc7_var = mfccs[6].var()
    mfcc8_mean = mfccs[7].mean()
    mfcc8_var = mfccs[7].var()
    mfcc9_mean = mfccs[8].mean()
    mfcc9_var = mfccs[8].var()
    mfcc10_mean = mfccs[9].mean()
    mfcc10_var = mfccs[9].var()
    mfcc11_mean = mfccs[10].mean()
    mfcc11_var = mfccs[10].var()
    mfcc12_mean = mfccs[11].mean()
    mfcc12_var = mfccs[11].var()
    mfcc13_mean = mfccs[12].mean()
    mfcc13_var = mfccs[12].var()
    mfcc14_mean = mfccs[13].mean()
    mfcc14_var = mfccs[13].var()
    mfcc15_mean =mfccs[14].mean()
    mfcc15_var = mfccs[14].var()
    mfcc16_mean =  mfccs[15].mean()
    mfcc16_var = mfccs[15].var()
    mfcc17_mean = mfccs[16].mean()
    mfcc17_var = mfccs[16].var()
    mfcc18_mean = mfccs[17].mean()
    mfcc18_var = mfccs[17].var()
    mfcc19_mean = mfccs[18].mean()
    mfcc19_var =mfccs[18].var()
    mfcc20_mean = mfccs[19].mean()
    mfcc20_var = mfccs[19].var()

    features = np.array([length,chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
                         spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean,
                         rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var ,harmony_mean, harmony_var,perceptr_mean, perceptr_var,
                         tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,
                         mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,
                         mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var
                         ,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,
                         mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var])
    return features

# create a function to classify an audio file
def classify_audio():
    filename = filedialog.askopenfilename()
    if filename:
        X = extract_features(filename)
        X = np.expand_dims(X, axis=0)# to fix input shape so it can be taken as input by model
        X=fit.transform(X)#transforemed  to the scale that we fitted above
        #print(X)#for debuggin remove later
        y_pred = model.predict(X)
        #print(y_pred)
        predicted_index=le.classes_[np.argmax(y_pred,axis=1)]
        #print(predicted_index)
        global predicted_genre
        predicted_genre=predicted_index[0]
        

        result_label = tk.Label(window, text='The genre for you music is',font=("Helvetica",20))
        result_label.pack()
        # create a label for displaying the classification result
        result_label = tk.Label(window, text=predicted_genre,font=("Helvetica",20))
        result_label.pack(pady=20)



# create a tkinter window
window = tk.Tk()
window.geometry("900x600")
window.title('Audio Genre Classification')


# create a file loading button
load_button = tk.Button(window, text='Load Audio File', command=classify_audio)#from here it goes to classify audio function
load_button.pack(pady=20)




# start the GUI
window.mainloop()