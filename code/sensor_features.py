from __future__ import division
import pandas as pd

DOOR_SENSOR_ID = 'a53'
DRAWER_SENSOR_IDS = ['a50', 'a51', 'a56']


def extract_features(df):
    """
    Extracts features from sensor data frame
    :param df: data frame with sensor data
    :return: data frame with extracted features
    """
    return get_features(df)


def get_features(df):
    some = []
    subject_no = df['subject_number'][0]
    i = subject_no
    while True:
        df_subject = df[df['subject_number'] == i]
        features_for_subject = features_for_single_subject(df_subject)
        features_for_subject['subject_number'] = int(i)
        some.append(features_for_subject)
        i += 1
        if df[df['subject_number'] == i].size == 0:
            break
    df_return = pd.DataFrame(some)
    df_return = df_return.fillna(0)
    return df_return


def features_for_single_subject(df):
    diffs = []
    diffs_per_sensor = {}
    start = 0
    for sensor in DRAWER_SENSOR_IDS:
        for index, row in df.iterrows():
            if row['sensor_id'] == sensor:
                if start == 0:
                    start = row['datetime']
                else:
                    x = (row['datetime'] - start).seconds
                    diffs.append(x)
                    start = 0
        if len(diffs) > 0:
            diffs_per_sensor[sensor] = diffs
        diffs = []
    dict_return = {}
    total_time_all_senors = 0
    total_activations_all_sensors = 0
    for key, value in diffs_per_sensor.items():
        dict_return['mean_time_' + key] = sum(value) / len(value)
        dict_return['max_time_' + key] = max(value)
        dict_return['min_time_' + key] = min(value)
        dict_return['total_time_' + key] = sum(value)
        total_time_all_senors += sum(value)
        dict_return['number_of_activations_' + key] = len(value)
        total_activations_all_sensors += len(value)
    for sensor in DRAWER_SENSOR_IDS:
        if sensor in diffs_per_sensor.keys():
            dict_return[sensor + '_time_over_total'] = dict_return['total_time_' + sensor] / total_time_all_senors
            dict_return[sensor + '_activations_over_total'] = \
                dict_return['number_of_activations_' + sensor] / total_activations_all_sensors
    return dict_return



