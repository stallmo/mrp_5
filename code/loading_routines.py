#import Dario's code (folder 'code_behaviorANDpersonality_masterProject/' needs to be in the code folder)
import sys
import time
sys.path.insert(0, '../code/code_behaviorANDpersonality_masterProject/') #eventually needs to adjusted

from data_extractor import *
import pandas as pd
import time as tm
import datetime
import glob

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
    columns = get_all_columns()
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
    
    #add column for task each frame belongs to        
    #task_timestamps = __get_task_timestamps(subject)
    #tasks = joints_df.apply(lambda row: __get_task(row, task_timestamps), axis=1)
    #joints_df['task'] = tasks
    
    return joints_df

def get_all_columns():
    return ['subject',
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

def __get_task_timestamps(subject):
    all_files_1 = glob.glob('../data/behavior_AND_personality_dataset/binary/*.txt')
    all_files_2 = glob.glob('../data/data_recordings_master/binary/*.txt')
    for f in all_files_1+all_files_2:
        if subject in f:
            return extract_tasks(f)

def __get_task(row, task_timestamps):
    row_time = datetime.datetime.strptime(str(row['time'].replace('\n', '')), "%a %b %d %H:%M:%S %Y").strftime('%H:%M:%S')
    #hour = dt.hour
    #minute = dt.minute
    #sec = dt.second
    #print hour, minute, sec
    
    for task_no, start_task in enumerate(task_timestamps):
        task_start_hour = start_task[3]
        task_start_minute = start_task[4]
        task_start_sec = start_task[5]
        
        start_task_time = datetime.datetime.strptime(str(task_start_hour)+':'+str(task_start_minute)+':'+str(task_start_sec), '%H:%M:%S').strftime('%H:%M:%S')

        #print 'row time: ', hour, minute, sec
        #print 'task_time: ', task_start_hour, task_start_minute, task_start_sec
        
        #if recording after start of current task
        if row_time>=start_task_time:
            #find first task which starts after recording of row
            for next_task_no, next_task_start in enumerate(task_timestamps[task_no:]):
                next_task_start_hour = next_task_start[3]
                next_task_start_minute = next_task_start[4]
                next_task_start_sec = next_task_start[5]
                
                next_task_time = datetime.datetime.strptime(str(next_task_start_hour)+':'+str(next_task_start_minute)+':'+str(next_task_start_sec), '%H:%M:%S').strftime('%H:%M:%S')

                if next_task_time>row_time:
                    return next_task_no-1
                
                if next_task_no==5:
                    return next_task_no
                
        return -10

def extract_tasks(filepath):
    '''
    :param filepath: path to txt file to read
    :return: a list with door entrance data (when subject entered the room to begin a task)
    '''


    f = open(filepath,'r')

    entrance = []

    #for every line in the file
    for i in f:
        tstamp = ''
    #if line is too short trash it
        if len(i) < 50:
            continue
    #get sensor id and status
        sensor = i[:3]
        if i[9:11] == 'ON':
            status = 'on'
        else:
            status = 'off'

    #after the long line of sensor name get timestamp
        for j in i[70:]:
            if j.isdigit():
                tstamp += (j)
    #if timestamp too short trash it
        if len(tstamp) < 12:
            continue

    #if some writing error occured in file clean it
        while tstamp[:2] in ('53','56','50','51'):
            tstamp = tstamp[2:]

    #take only hour,minute,second from timestamp
        tstamp = tstamp[6:]

    #convert string into actual timestamp
        ttstamp = time.strptime(tstamp[:2]+' '+tstamp[2:4]+' '+tstamp[4:6],'%H %M %S')

    #only consider door sensor on
    #TODO: can add other sensors here
    #NOTE: Other sensors are active when off (not linked)

        if sensor == 'a53' and status == 'on':
            entrance.append(ttstamp)


    #only consider when subject enters the room, not when exits (odd times, so even in the list)
    entrance = entrance[0::2]
    f.close()

    return entrance
