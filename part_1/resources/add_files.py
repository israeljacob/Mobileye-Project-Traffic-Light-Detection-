import os
import json
import shutil
from pathlib import Path

images_folder_path = Path('C://Users//yehuda//Documents//ExcelenTeam//Mobileye//Students FIrst Stage//Part_1 - Traffic ' \
                     'Light Detection - With Images//images_set')
destination_folder_path = Path('C://Users//yehuda//Documents//ExcelenTeam//Mobileye//Students FIrst Stage//Part_1 - ' \
                          'Traffic Light Detection - With Images//images_set//with_tfl')


#check if the destination folder exists, if not create it
if not os.path.exists(destination_folder_path):
    os.mkdir(destination_folder_path)

#copy all the images from the source folder to the destination folder if json file has traffic lights

for file_name in os.listdir(images_folder_path):
    if file_name.endswith('.json'):
        with open(os.path.join(images_folder_path, file_name), 'r') as json_file:
            json_data = json.load(json_file)
            for object in json_data['objects']:
                if object['label'] == 'traffic light':
                    shutil.copy(os.path.join(images_folder_path, file_name[:-21] + '_leftImg8bit.png'), destination_folder_path)
                    shutil.copy(file_name, destination_folder_path)
                    break
