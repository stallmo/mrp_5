import loading_routines
import feature_engineering
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt



def mov_amplitude(data, s_joint, m_joint):
    '''
    :param data: pandas df to extract feature from
    :param s_joint: still joint, ideal center of the movement
    :param m_joint: moving joint to get the amplitude of
    :return: pandas ds with frequency,max,min,mean of amplitudes
    '''
    data = preprocessing.remove_outliers(data)
    data = preprocessing.normalize_data(data)
    data = preprocessing.filter_noise(data,10)
    diff = abs(feature_engineering.calculate_joint_differences(data, only_for_columns=[ 'spineMid_x', 'spineMid_y', 'spineMid_z', 'handR_x', 'handR_y', 'handR_z']))

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

        if i[1]["spineMid_x-handR_x"]+_mean["spineMid_x-handR_x"] >=_max["spineMid_x-handR_x"]:
            c_x +=1
        if i[1]["spineMid_y-handR_y"]+_mean["spineMid_y-handR_y"] >=_max["spineMid_y-handR_y"]:
            c_y +=1
        if i[1]["spineMid_z-handR_z"]+_mean["spineMid_z-handR_z"] >=_max["spineMid_z-handR_z"]:
            c_z +=1

    #return creation
    ret = pd.DataFrame(columns=['X', 'Y', 'Z'])
    ret.loc[0] = [(float(c_x) / c),(float(c_y) / c),(float(c_z) / c)]
    ret.loc[1] = [_max["spineMid_x-handR_x"],_max["spineMid_y-handR_y"],_max["spineMid_z-handR_z"]]
    ret.loc[2] = [_min["spineMid_x-handR_x"], _min["spineMid_y-handR_y"], _min["spineMid_z-handR_z"]]
    ret.loc[3] = [_mean["spineMid_x-handR_x"], _mean["spineMid_y-handR_y"], _mean["spineMid_z-handR_z"]]

    ''' plot creation if wanted
    plt.plot(range(len(diff['spineMid_x-handR_x'])), diff['spineMid_x-handR_x'])
    plt.grid(True)
    plt.show()
    '''

    return ret



