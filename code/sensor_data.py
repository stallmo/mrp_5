import pandas as pd
import os
import re
from datetime import datetime, time
import os.path
import time as tm
import split_tasks
import glob
import sensor_features


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
    # Correct subject 9 txt file:
    correct_subject9_file(path_to_dir)
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
    current_subject = rows[0][0]
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

        if row == rows[len(rows) - 1]:
            data_all_subjects.append(data)

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


def correct_subject9_file(path_to_dir):
    # fix error in data: subject 9, line 53, sensor a50, status ON, 17-02-55 => 17-04-55
    if os.path.isfile(path_to_dir + '18-10-16_sensors_subject9.txt'):
        s = open(path_to_dir + '18-10-16_sensors_subject9.txt').read()
        s = s.replace('17-02-55', '17-04-55')
        f = open(path_to_dir + '18-10-16_sensors_subject9.txt', 'w')
        f.write(s)
        f.close()


def sensor_data_per_task(path, task_number=1):
    """
    binary sensor dataframe for all subject for specific task
    :param path: directory where txt files with binary sensor data are
    :param task_number: can be either 1, 2, 3, 4, 5 or 6
    :return:
    """
    df = sensor_data_to_data_frame(path)
    df_per_subject = []
    current_subject = df['subject_number'][0]
    lst = []
    for index, row in df.iterrows():
        if current_subject == row['subject_number']:
            lst.append(row)
        else:
            df_per_subject.append(lst)
            lst = []
            current_subject = row['subject_number']
            lst.append(row)
        if index == df.count()['subject_number'] - 1:
            df_per_subject.append(lst)

    paths_to_txt_files = glob.glob(path + '*.txt')
    task_times = {}
    for path in paths_to_txt_files:
        task_times[int(re.findall('subject?(\d+)', path)[0])] = split_tasks.extract_tasks(path)

    if len(df_per_subject) != len(task_times):
        return "Task start times and number of subject not equal!"

    timings = []
    for key, value in task_times.iteritems():
        timings.append(value)

    rows_in_time_frame_task = []
    for i in range(len(timings)):
        for row in df_per_subject[i]:
            x = datetime.time(row['datetime'])
            y = time(timings[i][0][3], timings[i][0][4], timings[i][0][5])
            if task_number == 6:
                if time(timings[i][5][3], timings[i][5][4], timings[i][5][5]) > datetime.time(row['datetime']):
                    rows_in_time_frame_task.append(row)
            else:
                if time(timings[i][task_number-1][3], timings[i][task_number-1][4], timings[i][task_number-1][5]) \
                        < datetime.time(row['datetime']) < \
                        time(timings[i][task_number][3], timings[i][task_number][4], timings[i][task_number][5]):
                    rows_in_time_frame_task.append(row)

    df_result = pd.DataFrame(rows_in_time_frame_task)
    return df_result.reset_index()


def main():
    path = '../data/behavior_AND_personality_dataset/binary/'
    path2 = '../data/data_recordings_master/binary/'
    # df = sensor_data_to_data_frame(path)
    # df2 = sensor_data_to_data_frame('../data/data_recordings_master/binary/')
    # print df
    # print df2
    # print sensor_features.extract_features(df)
    # print sensor_features.extract_features(df2)
    # df = sensor_data_per_task(path, 6)
    # print sensor_features.extract_features(df)
    # df2 = sensor_data_per_task(path2, 4)
    # print sensor_features.extract_features(df2)


class BinarySensorData:
    def __init__(self):
        self.subject = None
        self.a50 = []
        self.a51 = []
        self.a56 = []


if __name__ == '__main__':
    main()
