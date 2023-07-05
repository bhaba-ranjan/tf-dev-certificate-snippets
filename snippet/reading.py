import csv
import numpy as np
from PIL import Image


# reading csv file for text data

def parse_data_from_csv(file_path, reshape=(None, None)):
    with open(file_path) as file:
        labels = []
        data = []

        csv_reader = csv.reader(file, delimiter='\n')
        print(next(csv_reader))
        for row in csv_reader:
            try:
                row = row[0].split(',')
                labels.append(int(row[-1]))
                data.append(row[-2])
            except:
                print(f' exception during: {row}')

        labels = np.array(labels, dtype=np.float32)
        data = np.array(data)

        return data, labels


# Reading from csv for images

def parse_image(dataset, reshape=(None, None)):
    data = []
    labels = []

    for img_raw, label in dataset:
        img = Image.fromarray(img_raw.numpy())
        img_reshaped = img.resize(reshape)
        data.append(np.array(img_reshaped, dtype=np.float64))
        labels.append(label.numpy())

    data = np.array(data, dtype=np.float64)
    labels = np.array(labels, dtype=np.float32)

    return data, labels


# Reading from json file

import json


def parse_json(file_path):
    headLine = []
    label = []
    lengths = []
    with open(file_path) as file:
        for jsonObject in file:
            parsedObject = json.loads(jsonObject)
            # Get required property from parsed object
            headLine.append(parsedObject['headline'])
            lengths.append(len(headLine[-1].split(' ')))
            label.append(parsedObject['is_sarcastic'])

    print(f'headlines: {len(headLine)}')
    print(f'labels: {len(label)}')
    print(f'maximum length of string by WORDS: {max(lengths)}')

    return headLine, label, max(lengths)
