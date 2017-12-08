import pandas as pd
from math import pow, sqrt
import preprocessing




def speed(data_frame,joint="head", start=0, end=None, step=30):
    """
    Calculates the average speed of the head
    :param data_frame: pandas dataframe.
    :param start: frame to start the computation.
    :param end: frame to end the computation.
    :param step: use frame only every step.
    :return: average speed of the head per frame
    """

    # Pre-processing
    data_frame = preprocessing.ewma_noise_filter(data_frame)
    data_frame = preprocessing.recalculate_joint_positions(data_frame, joint)
    
    subject = data_frame.iloc[0]['subject']


    # TODO check arguments for validity
    x_label = joint + '_real_x'
    y_label = joint + '_real_y'
    z_label = joint + '_real_z'


    lastx = data_frame.iloc[0][x_label]
    lasty = data_frame.iloc[0][y_label]
    lastz = data_frame.iloc[0][z_label]
    if(end == None):
        stop = len(data_frame)
    else:
        stop = end


    speeds = []
    # TODO: Add speed for first frame. Interpolate or add default
    for index in range(start + 1, stop, step):
        row = data_frame.iloc[index]
        x = row[x_label]
        y = row[y_label]
        z = row[z_label]

        # calculate distance (Euclidian}
        dist = sqrt((x-lastx)**2 + (y-lasty)**2 + (z-lastz)**2)
        # calculate speed
        velocity = dist/step
        speeds.append(velocity * 100)

        lastx = x
        lasty = y
        lastz = z

    speeds_df = pd.DataFrame(speeds)
    mean_speed = speeds_df.mean()[0]
    min_speed = speeds_df.min()[0]
    max_speed = speeds_df.max()[0]
    median_speed = speeds_df.median()[0]

    return pd.DataFrame({'subject':[subject], 'mean_speed': [mean_speed], 'median_speed': [median_speed], 'min_speed':[min_speed], 'max_speed': [max_speed]})