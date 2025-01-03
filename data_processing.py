import numpy as np
import librosa 
import os

import pandas as pd  

#Path to the dataset
DATA_PATH = '/Users/dangriffith/Library/CloudStorage/OneDrive-CarletonUniversity/Resume Projects/Marine_Mammal_Identifier/data'



def load_sound_file(file_name): 

    y, sr = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

def process_class(file_path, label):
    data = []
    csv_file_path = r'marine_mammal_mfccs'
    class_data = os.listdir(file_path)

    for data_point in class_data:
        file = os.path.join(file_path, data_point)
        mfcc = load_sound_file(file)
        data.append({'vecotor': mfcc, 'label': label})
    
    df = pd.DataFrame(data)
    write_header = not os.path.exists(csv_file_path)    
    df.to_csv(csv_file_path, mode='a', index=False, header=write_header)
    return f'Finished Processing {len(df)}'
        


def process_data(DATA_PATH = DATA_PATH):
    print(DATA_PATH)
    classes = os.listdir(DATA_PATH)
    for data_point in classes:
        file = os.path.join(DATA_PATH,data_point)
        label = data_point
        print(label)
        print(file)
        if 'DS_Store' not in file:
            process_class(file, label)

if __name__ == '__main__':

    print(process_data())




