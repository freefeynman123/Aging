import os

ORIGIN_DIR = 'data/UTKFace'
DESTINATION_DIR = 'data/renamed'

def rename_files():
    for file in os.listdir(ORIGIN_DIR)
        #retrieving age and gender from file names
        splitted = file.split('_')
        age = int(splitted[0])
        gender = int(splitted[1])
        destination_name = '.'.join(['_'.join([age, gender]), 'jpg'])
        os.rename(os.path.join(ORIGIN_DIR, file), os.path.join(DESTINATION_DIR, destination_name))

def convert_age(age: int):

