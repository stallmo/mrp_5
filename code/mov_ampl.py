import loading_routines
import feature_engineering
import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def mov_amplitude(data, s_joint, m_joint):
    '''
    :param data: pandas df to extract feature from
    :param s_joint: still joint, ideal center of the movement
    :param m_joint: moving joint to get the amplitude of
    :return: pandas ds with frequency,max,min,mean of amplitudes
    '''
    s_coords = [s_joint+'_x', s_joint+'_y', s_joint+'_z']
    m_coords = [m_joint + '_x', m_joint + '_y', m_joint + '_z']



    delta_x = s_coords[0]+ "-" + m_coords[0]
    delta_y = s_coords[1] + "-"+m_coords[1]
    delta_z = s_coords[2] + "-"+m_coords[2]


    data = preprocessing.remove_outliers(data)
    data = preprocessing.normalize_data(data)
    data = preprocessing.filter_noise(data,10)
    diff = abs(feature_engineering.calculate_joint_differences(data, only_for_columns=[s_coords[0],s_coords[1],s_coords[2],m_coords[0],m_coords[1],m_coords[2]]))

    #get min max and mean of the differences
    _max = diff.max()
    _min = diff.min()
    _mean = diff.mean()

    #counters for frequency
    c = 0
    c_x = 0
    c_y = 0
    c_z = 0

    #cycle to detect frequency of "ample" actions given the max-mean threshold
    for i in diff.iterrows():
        c+=1

        if i[1][delta_x]+_mean[delta_x] >=_max[delta_x]:
            c_x +=1
        if i[1][delta_y]+_mean[delta_y] >=_max[delta_y]:
            c_y +=1
        if i[1][delta_z]+_mean[delta_z] >=_max[delta_z]:
            c_z +=1

    #return creation
    ret = pd.DataFrame(columns=['ampl_freq_'+delta_x, 'ampl_freq_'+delta_y, 'ampl_freq_'+delta_z,'ampl_max_'+delta_x,'ampl_max_'+delta_y,'ampl_max_'+delta_z,'ampl_min_'+delta_x,'ampl_min_'+delta_y,'ampl_min_'+delta_z,'ampl_mean_'+delta_x,'ampl_mean_'+delta_y,'ampl_mean_'+delta_z])
    ret.loc[s_joint+'-'+m_joint] = [(float(c_x) / c),(float(c_y) / c),(float(c_z) / c),_max[delta_x],_max[delta_y],_max[delta_z],_min[delta_x], _min[delta_y], _min[delta_z],_mean[delta_x], _mean[delta_y], _mean[delta_z]]


    ''' plot creation if wanted
    plt.plot(range(len(diff['spineMid_x-handR_x'])), diff['spineMid_x-handR_x'])
    plt.grid(True)
    plt.show()
    '''

    return ret
