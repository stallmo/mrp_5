import pandas as pd
import feature_engineering.py
from math import pow, sqrt
from statistics import mean

def speed(data_frames : pd.DataFrame, start=0, end=None, step=1):
    """
    Calculates the average speed of the head
    :param data_frames: pandas dataframe.
    :param start: frame to start the computation.
    :param end: frame to end the computation.
    :param step: use frame only every step.
    :return: average speed of the head per frame
    """

    # TODO check arguments for validity
    x_label = 'head_x'
    y_label = 'head_y'
    z_label = 'head_z'


    lastx = data_frames[0][x_label]
    lasty = data_frames[0][y_label]
    lastz = data_frames[0][z_label]
    if(end == None):
        stop = len(data_frames)
    else:
        stop = end


    speeds = []
    for index in range(start + 1, stop, step):
        row = data_frames[index]
        x = row[x_label]
        y = row[y_label]
        z = row[z_label]

        # calculate distance (Euclidian}
        dist = sqrt((x-lastx)**2 + (y-lasty)**2 + (z-lastz)**2)
        # calculate speed
        velocity = dist/step
        speeds.append(velocity)

        lastx = x
        lasty = y
        lastz = z
        
    return mean(speeds)

if __name__ == "__main__":
    loading_routines.load_df_from_xml("data/")