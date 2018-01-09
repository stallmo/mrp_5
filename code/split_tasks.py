import loading_routines
import feature_engineering
import time as tm
import pandas as pd





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
        ttstamp = tm.strptime(tstamp[:2]+' '+tstamp[2:4]+' '+tstamp[4:6],'%H %M %S')

    #only consider door sensor on
    #TODO: can add other sensors here
    #NOTE: Other sensors are active when off (not linked)

        if sensor == 'a53' and status == 'on':
            entrance.append(ttstamp)


    #only consider when subject enters the room, not when exits (odd times, so even in the list)
    entrance = entrance[0::2]

    return entrance

def split_tasks(data, subject, path_to_sensor):

    '''
    :param data: pandas df for a subject
    :param subject: subject number as a string
    :return: 6tuple of dataframes with data per task
    '''

    #tasks = extract_tasks('../data/data_recordings_master/binary/18-10-16_sensors_'+subject+'.txt')
    tasks = extract_tasks(path_to_sensor)
    

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
               'kneeR_x', 'kneeR_y', 'kneeR_z',
               'ankleR_x', 'ankleR_y', 'ankleR_z',
               'footR_x', 'footR_y', 'footR_z',
               'hipL_x', 'hipL_y', 'hipL_z',
               'kneeL_x', 'kneeL_y', 'kneeL_z',
               'ankleL_x', 'ankleL_y', 'ankleL_z',
               'footL_x', 'footL_y', 'footL_z']

    task0 = []
    task1 = []
    task2 = []
    task3 = []
    task4 = []
    task5 = []
    all_tasks = []

    for i,r in data.iterrows():

        dtime = tm.strptime(r['time'][11:19],'%H:%M:%S' )

        if dtime > tasks[5]:
            task5.append(r)

        if (dtime < tasks[5]) and (dtime > tasks[4]):
            task4.append(r)

        if (dtime < tasks[4]) and (dtime > tasks[3]):
            task3.append(r)

        if (dtime < tasks[3]) and (dtime > tasks[2]):
            task2.append(r)

        if (dtime < tasks[2]) and (dtime > tasks[1]):
            task1.append(r)

        if (dtime < tasks[1]) and (dtime > tasks[0]):
            task0.append(r)

    t0 = pd.DataFrame(task0, columns=columns)
    t1 = pd.DataFrame(task1, columns=columns)
    t2 = pd.DataFrame(task2, columns=columns)
    t3 = pd.DataFrame(task3, columns=columns)
    t4 = pd.DataFrame(task4, columns=columns)
    t5 = pd.DataFrame(task5, columns=columns)

    return t0,t1,t2,t3,t4,t5

def get_task_column(data, path_to_sensor):
    
    tasks = extract_tasks(path_to_sensor)
    
    task_column = []
    
    for i,r in data.iterrows():

        dtime = tm.strptime(r['time'][11:19],'%H:%M:%S' )

        if dtime > tasks[5]:
            task_column.append(5)

        if (dtime < tasks[5]) and (dtime > tasks[4]):
            task_column.append(4)

        if (dtime < tasks[4]) and (dtime > tasks[3]):
            task_column.append(3)

        if (dtime < tasks[3]) and (dtime > tasks[2]):
            task_column.append(2)

        if (dtime < tasks[2]) and (dtime > tasks[1]):
            task_column.append(1)

        if (dtime < tasks[1]) and (dtime > tasks[0]):
            task_column.append(0)
    
    return task_column
    
#%%
    """
import glob
all_subjects = ['subject'+str(i) for i in range(21,48)]
all_subjects = all_subjects+['subject50']
subject_frames_files = glob.glob('../data/data_recordings_master/joints/*.xml')
all_subjects_dfs = [loading_routines.load_df_from_xml(f) for f in subject_frames_files]
#%%
failed = []
for subject_df in all_subjects_dfs:
    subject = subject_df['subject'][0]
    print subject
    tasks = split_tasks(subject_df, subject)
    task_sum = 0
    for i, task in enumerate(tasks):
        if len(task)==0:
          failed.append(subject)
        task_sum += len(task)
        print 'task: ',i,' length of task dfs: ', len(task)
    print 'total frames in original dataframe: ', len(subject_df), 'total frames by task: ', task_sum
    if len(subject_df)!=task_sum:
        print 'Total frames per subject and total task frames do not match!'
#%%
failed = []
for subject_df in all_subjects_dfs:
    subject = subject_df['subject'][0]
    path = '../data/data_recordings_master/binary/18-10-16_sensors_'+subject+'.txt'
    col = get_task_column(subject_df, path)
    print subject
    print 'len task column: ', len(col), ' len dataframe: ', len(subject_df)
    if len(col)!=len(subject_df):
        print 'Alarm!'
        failed.append(subject)"""