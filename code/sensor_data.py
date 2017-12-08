import pandas as pd
import os
import re
from datetime import datetime
from sensor_features import extract_features


#####################################################
# Put sensor data from text files into data frame
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
        split_line = line.split()
        if 'BUTTON' in line and len(split_line) > 2:
            subject_number = int(re.findall('subject?(\d+)', file_name)[0])
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
    """
    Only leaves rows with two subsequent entries for each sensor
    Every sensor activation should consist of exactly one OFF and one ON row
    :param rows: rows with more (or less) than 2 entries per sensor activation
    :return: cleaned rows
    """
    cleaned_rows = []
    current_sensor = rows[0][1]
    temp_row_memory = []
    for row in rows:
        if row[1] == current_sensor:
            temp_row_memory.append(row)
        else:
            if len(temp_row_memory) > 1:
                cleaned_rows += temp_row_memory[-2:]
            temp_row_memory = [row]
            current_sensor = row[1]
    return cleaned_rows


def main():
    path = '../data/behavior_AND_personality_dataset/binary/'
    df = sensor_data_to_data_frame(path)
    # print(df)
    print(extract_features(df))


if __name__ == '__main__':
    main()
