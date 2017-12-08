import pandas as pd
import os
import re
from datetime import datetime


#####################################################
#####################################################
# Get sensor data from text files into data frame
#####################################################
#####################################################
def sensor_data_to_data_frame(path_to_dir):
    """
    Returns a data frame with sensors information for every line in txt file, row order:
    subject_number, sensor_id, status, date, time
    :param path_to_dir:
    :return:
    """
    rows = []
    # Get sorted file names:
    file_names = get_sorted_file_names(path_to_dir)
    # For each file/subject extract the features
    for file_name in file_names:
        file = open(path_to_dir + file_name, 'r')
        rows += extract_lines_data(file, file_name)
    cleaned_rows = clean_rows(rows)
    # Put columns and rows into a data frame:
    df = pd.DataFrame(cleaned_rows, columns=get_column_names())
    return df


def get_column_names():
    return ['subject_number', 'sensor_id', 'status', 'datetime']


def extract_lines_data(file, file_name):
    lines = []
    for line in file.readlines():
        if 'BUTTON' in line:
            subject_number = int(re.findall('subject?(\d+)', file_name)[0])
            split_line = line.split()
            sensor_id = re.findall('a5\d', split_line[0])[0]
            status = re.findall('BUTTON?([A-Z]+)', split_line[0])[0]
            date = split_line[1].split('-')
            time = split_line[2].split('-')
            dt = datetime(int('20' + date[0]), int(date[1]), int(date[2]),
                          int(time[0]), int(time[1]), int(time[2]), 0)
            lines.append([subject_number, sensor_id, status, dt])
    return lines


def get_sorted_file_names(path_to_dir):
    file_names = []
    for file in os.listdir(path_to_dir):
        if file.endswith(".txt"):
            file_names.append(file)
    # put file names in correct order:
    file_names.sort(key=lambda x: int(re.findall('subject?(\d+)|$', x)[0]))
    return file_names


def clean_rows(rows):
    cleaned_rows = []
    current_sensor = rows[0][1]
    temp_row_memory = []
    for row in rows:
        if row[1] == current_sensor:
            temp_row_memory.append(row)
        else:
            if len(temp_row_memory) > 1:
                cleaned_rows += temp_row_memory[-2:]
            temp_row_memory = []
            current_sensor = row[1]
    return cleaned_rows


#####################################################
#####################################################
# Extract features from data frame
#####################################################
#####################################################
def extract_door_opening(data):
    door_sensor_activations = []
    temp = []
    for element in data:
        # Sensor Id 'a53' is the door
        if element[0] == 'a53':
            temp.append(element[3])
        elif len(temp) > 1:
            door_sensor_activations.append(temp[-2])
            door_sensor_activations.append(temp[-1])
            temp = []
    while len(door_sensor_activations) != len(door_column_names()):
        door_sensor_activations.append(None)
    return door_sensor_activations


def mean_time_open(data):
    time_diffs = []
    max_diff = 0
    min_diff = 0
    for i in range(len(data) - 1):
        diff = (data[i + 1][3] - data[i][3]).seconds
        if i == 0:
            min_diff = diff
        elif diff < min_diff:
            min_diff = diff
        if diff > max_diff:
            max_diff = diff
        time_diffs.append(diff)
    if len(time_diffs)==0:
        mean_sen = 0
    else:
        mean_sen = sum(time_diffs) / len(time_diffs)
    return {'mean_time_sensor_' + data[0][0]: mean_sen,
            'max_time_sensor_' + data[0][0]: max_diff,
            'min_time_sensor_' + data[0][0]: min_diff,
            'total_time_sensor_' + data[0][0]: sum(time_diffs)}


def extract_sensor(sensor, data):
    rlist = []
    for element in data:
        if element[0] == sensor:
            rlist.append(element)
    return rlist


def clean_line_data(data):
    """
    Cleans the data so that for each sensor there are actually two entries for each activation
    :param data:
    :return: a cleaned data list
    """
    cleaned_data = []
    temp = data[0][0]
    data_per_sensor = []
    for element in data:
        sensor = element[0]
        if sensor == temp:
            data_per_sensor.append(element)
        else:
            if len(data_per_sensor) > 1:
                cleaned_data.append(data_per_sensor[-2])
                cleaned_data.append(data_per_sensor[-1])
            temp = element[0]
            data_per_sensor = [element]
    print(cleaned_data)
    return cleaned_data


def extract_features(file, file_name):
    extracted_features = {}
    data = extract_lines_data(file, file_name)
    data = clean_line_data(data)
    for sensor in ['a50', 'a51', 'a56']:
        sensor_specific_data = extract_sensor(sensor, data)
        # Order of feature extraction method calls important!
        extracted_features = extracted_features + mean_time_open(sensor_specific_data)
    # return subject number with extracted features:
    return re.findall('subject?(\d+)', file_name) + extracted_features


def door_column_names():
    column_names = []
    for i in range(1, 7):
        column_names.append("door_open_" + str(i))
        column_names.append("door_close_" + str(i))
    return column_names


def main():
    path = '../data/behavior_AND_personality_dataset/binary/'
    df = sensor_data_to_data_frame(path)
    print(df)
    print(df[df['subject_number'] == 3])


if __name__ == '__main__':
    main()
