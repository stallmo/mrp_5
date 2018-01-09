import pandas as pd
import os
import re
from datetime import datetime
from sensor_features import extract_features
from loading_routines import extract_tasks


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
        f = open(path_to_dir + file_name, 'r')
        rows += extract_lines_data(f, file_name)
    cleaned_rows = clean_rows(rows)
    # Put columns and rows into a data frame:
    df = pd.DataFrame(cleaned_rows, columns=get_column_names())
    return df


def get_column_names():
    return ['subject_number', 'sensor_id', 'status', 'datetime']


def extract_lines_data(txt_file, file_name):
    lines = []
    for line in txt_file.readlines():
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
    for file_name in os.listdir(path_to_dir):
        if file_name.endswith(".txt"):
            file_names.append(file_name)
    # put file names in correct order:
    file_names.sort(key=lambda x: int(re.findall('subject?(\d+)|$', x)[0]))
    return file_names


def clean_rows(rows):
    """
    Cleans the rows
    :param rows: rows
    :return: cleaned rows
    """
    cleaned_rows = []
    current_sensor = rows[0][1]
    current_subject = rows[0][0]
    temp_row_memory = []
    data_all_subjects = []
    data = BinarySensorData()
    data.subject = current_subject
    for row in rows:
        if row[0] != current_subject:
            current_subject = row[0]
            data_all_subjects.append(data)
            data = BinarySensorData()
            data.subject = current_subject

        if row[1] == 'a50':
            data.a50.append(row)
        elif row[1] == 'a51':
            data.a51.append(row)
        elif row[1] == 'a56':
            data.a56.append(row)

        if row[1] == current_sensor:
            temp_row_memory.append(row)
        else:
            if len(temp_row_memory) > 1:
                cleaned_rows += temp_row_memory[-2:]
            temp_row_memory = [row]
            current_sensor = row[1]

    for data in data_all_subjects:
        if len(data.a50) % 2 != 0:
            prev_status = ''
            current_status = ''
            indices_to_remove = []
            for i, v in enumerate(data.a50):
                if current_status == '':
                    current_status = v[2]
                else:
                    prev_status = current_status
                    current_status = v[2]
                if current_status == prev_status:
                    indices_to_remove.append(i)
                elif i == 0 and v[2] == 'ON':
                    indices_to_remove.append(i)
                elif i == len(data.a50) - 1 and v[2] == 'OFF':
                    indices_to_remove.append(i)
            if len(indices_to_remove) != 0:
                indices_to_remove.reverse()
            for i in indices_to_remove:
                del data.a50[i]
        if len(data.a51) % 2 != 0:
            prev_status = ''
            current_status = ''
            indices_to_remove = []
            for i, v in enumerate(data.a51):
                if current_status == '':
                    current_status = v[2]
                else:
                    prev_status = current_status
                    current_status = v[2]
                if current_status == prev_status:
                    indices_to_remove.append(i)
                elif i == 0 and v[2] == 'ON':
                    indices_to_remove.append(i)
                elif i == len(data.a51) - 1 and v[2] == 'OFF':
                    indices_to_remove.append(i)
            if len(indices_to_remove) != 0:
                indices_to_remove.reverse()
            for i in indices_to_remove:
                del data.a51[i]
        if len(data.a56) % 2 != 0:
            prev_status = ''
            current_status = ''
            indices_to_remove = []
            for i, v in enumerate(data.a56):
                if current_status == '':
                    current_status = v[2]
                else:
                    prev_status = current_status
                    current_status = v[2]
                if current_status == prev_status:
                    indices_to_remove.append(i)
                elif i == 0 and v[2] == 'ON':
                    indices_to_remove.append(i)
                elif i == len(data.a56) - 1 and v[2] == 'OFF':
                    indices_to_remove.append(i)
            if len(indices_to_remove) != 0:
                indices_to_remove.reverse()
            for i in indices_to_remove:
                del data.a56[i]

    crs = []
    for data in data_all_subjects:
        for i in data.a50:
            crs.append(i)
        for i in data.a51:
            crs.append(i)
        for i in data.a56:
            crs.append(i)

    return crs


def main():
    path = '../data/behavior_AND_personality_dataset/binary/'
    extract_tasks(path + '18-10-16_sensors_subject37.txt')
    df = sensor_data_to_data_frame(path)
    # print df
    # print extract_features(df)


class BinarySensorData:
    def __init__(self):
        self.subject = None
        self.a50 = []
        self.a51 = []
        self.a56 = []


if __name__ == '__main__':
    main()
