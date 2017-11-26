import loading_routines
import feature_engineering
import preprocessing
import matplotlib.pyplot as plt



def mov_amplitude(data, s_joint, m_joint):
    '''
    :param data: pandas df to extract feature from
    :param s_joint: still joint, ideal center of the movement
    :param m_joint: moving joint to get the amplitude of
    :return: 3x3 list of tuples, with x,y,z for min,max,avg
    '''
    ret = []

    data = preprocessing.remove_outliers(data)
    data = preprocessing.normalize_data(data)
    data = preprocessing.filter_noise(data, 10)
    diff = abs(feature_engineering.calculate_joint_differences(data, only_for_columns=[s_joint+'_x', s_joint+'_y',
                                                                                       s_joint+'_z', m_joint+'_x',m_joint+'_y',m_joint+'_z']))

    ret.append((diff.max().tolist(), diff.min().tolist(), diff.mean().tolist()))
    return ret



