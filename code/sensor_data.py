import pandas as pd
import os
import re


def sensor_data_to_data_frame(path_to_dir):
    """
    Returns a data frame with sensors information for every row, row order:
    subject_number, sensor_id, status, date, time
    :param path_to_dir:
    :return:
    """
    rows = []
    # Get sorted file names:
    file_names = get_sorted_file_names(path_to_dir)
    # Fill rows with data from files:
    for name in file_names:
        file = open(path_to_dir + name, 'r')
        for line in file.readlines():
            rows.append(get_data_from_line(line, name))
    # Put columns and rows into a data frame:
    df = pd.DataFrame(rows, columns=get_column_names())
    return df


def get_data_from_line(line, file_name):
    if 'BUTTON' in line:
        sl = line.split()
        # sometimes a line has an extra date or time stamp (do not know why):
        if len(sl) > 3:
            sl = sl[:-1]
        # getting the id of the sensor:
        sensor_id = re.findall('a5\d', sl[0])[0]
        # stripping first element of split down to just ON or OFF:
        sl[0] = re.findall('BUTTON?([A-Z]+)', sl[0])[0]
        # putting it all together: (order= subject_number,id,ON/OFF,date,time)
        return re.findall('subject?(\d+)', file_name) + [sensor_id] + sl


def get_sorted_file_names(path_to_dir):
    file_names = []
    for file in os.listdir(path_to_dir):
        if file.endswith(".txt"):
            file_names.append(file)
    # put file names in correct order:
    file_names.sort(key=lambda x: int(re.findall('subject?(\d+)|$', x)[0]))
    return file_names


def get_column_names():
    return ['subject_number', 'sensor_id', 'status', 'date', 'time']


def main():
    path = '../data/behavior_AND_personality_dataset/binary/'
    print(sensor_data_to_data_frame(path))


if __name__ == '__main__':
    main()
