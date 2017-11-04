#import Dario's code (folder 'code_behaviorANDpersonality_masterProject/' needs to be in the code folder)
import sys
sys.path.insert(0, './code/code_behaviorANDpersonality_masterProject')

from data_extractor import *
import pandas as pd

def load_df_from_xml(path_to_xml, n_joints=21):
    """
    :type path_to_xml: string
    :param path_to_xml: path to the xml file that contains the body joint data to be loaded
    :type n_joints: integer
    :param n_joints: number of joints
    :rtype: pandas dataframe
    :rparam: pandas dataframe with all bodyjoints per frame (one frame per row)
    """
    parsed_xml = xml_parser(path_to_xml)
    #find subject
    pos1 = path_to_xml.find('/subject')+1
    pos2 = path_to_xml.find('_points')
    subject = path_to_xml[pos1:pos2]
    columns = ['subject',
                'frameId', 'time', 'trackingId',
               'head_x', 'head_y', 'head_z',
              'neck_x', 'neck_y', 'neck_z',
              'spineMid_x', 'spineMid_y', 'spineMid_z',
              'spineBase_x', 'spineBase_y', 'spineBase_z',
              'spineShoulder_x', 'spineShoulder_y', 'spineShoulder_z',
              'shoulderR_x', 'shoulderR_y', 'shoulderR_z',
              'elbowR_x', 'elbowR_y', 'elbowR_z',
              'wristR_x', 'wristR_y', 'wristR_z',
              'handR_x', 'handR_y', 'handR_z',
              'shoulderL_x', 'shoulderL_y', 'shoulderL_z',
              'elbowL_x', 'elbowL_y', 'elbowL_z',
              'wristL_x', 'wristL_y', 'wristL_z',
              'handL_x', 'handL_y', 'handL_z',
              'hipR_x', 'hipR_y', 'hipR_z',
              'kneeR_x', 'kneeR_y','kneeR_z',
              'ankleR_x', 'ankleR_y', 'ankleR_z',
              'footR_x', 'footR_y', 'footR_z',
              'hipL_x', 'hipL_y', 'hipL_z',
              'kneeL_x', 'kneeL_y', 'kneeL_z',
              'ankleL_x', 'ankleL_y', 'ankleL_z',
              'footL_x', 'footL_y', 'footL_z']
    #n_joints = 21
    first_joint_no = 1 # element no in parsed list
    rows = []
    for df_ind, frame in enumerate(parsed_xml):
        #add subject
        row = [subject]
        #add trackinfo
        for i in range(3):
            row.append(frame[0][i])
        #add joints
        for joint in range(first_joint_no,n_joints+first_joint_no):
            for coord in range(3):
                row.append(float(frame[joint][coord]))
        #print(len(row))
        if df_ind % 1000==0:
            print('Loaded {0} tracks for "{1}"'.format(df_ind, subject))
        rows.append(row)

    joints_df = pd.DataFrame(rows, columns=columns)
    return joints_df
