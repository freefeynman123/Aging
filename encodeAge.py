import os
import shutil
from typing import List

ORIGIN_DIR = 'data/UTKFace'
DESTINATION_DIR = 'data/renamed'

def rename_files():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    else:
        x = input("Do you want to rewrite the folder? Y/N").lower()
        if x == 'y':
            shutil.rmtree(DESTINATION_DIR, ignore_errors=True)
            os.makedirs(DESTINATION_DIR)
        elif x == 'n':
            print("You chose not to delete the folder")
        else:
            print("Please give Y/N response")
    for index, file in enumerate(os.listdir(ORIGIN_DIR)):
        print(ORIGIN_DIR)
        #retrieving age and gender from file names
        splitted = file.split('_')
        age = str(splitted[0])
        gender = str(splitted[1])
        destination_name = '.'.join(['_'.join([age, gender, str(index)]), 'jpg'])
        os.rename(os.path.join(ORIGIN_DIR, file), os.path.join(DESTINATION_DIR, destination_name))


def convert_age(age: int, interval: List[int]):
    for index, value in enumerate(interval):
        if age <= value:
            return index

rename_files()



