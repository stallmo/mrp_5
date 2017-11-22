from loading_routines import load_df_from_xml
import os
import re
from features.speed_feature import speed
from pandas import DataFrame
#from pickle import Pickler, Unpickler
from cPickle import Pickler,Unpickler

def getsubject_name(pathname, expression):
    return expression.match(pathname).group(0)


def main():
    print "current dir = {}".format(os.getcwd())
    path_to_data = "C:\Users\Jan\PycharmProjects\mrp_5\code\data\joints"
    path_to_pickle = "C:\Users\Jan\PycharmProjects\mrp_5\code\data\data.p"
    path_format = "C:\Users\Jan\PycharmProjects\mrp_5\code\data\joints\{}_points.xml"
    # read data
    # Check if pickled data exists
    expr = re.compile("subject[0-9]+")
    subjects = [getsubject_name(x, expr) for x in os.listdir(path_to_data)]
    try:
        pickleFile = open(path_to_pickle)
        unpickler = Unpickler(pickleFile)
        print"Loading from pickle"
        data = unpickler.load()
    except IOError:
        print "No pickle file found. reloading from xml"
        #reload all data
        data = dict()
        """:type : dict(str, DataFrame)"""
        for subject in subjects:
            subject_path = path_format.format(subject)
            try:
                data[subject] = load_df_from_xml(subject_path)
            except IOError as e:
                print "skipping {}, because {}".format(subject_path, e)
        pickleFile = open(path_to_pickle, 'w')
        pick = Pickler(pickleFile)
        pick.dump(data)
    # apply preprocessing
    print "Applying preprocessing steps"

    # extract features
    print "Extracting features"
    x = speed(data[subjects[0]])
    pass

    #save data as moi

if __name__ == '__main__':
    main()