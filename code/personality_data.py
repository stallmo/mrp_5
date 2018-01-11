import pandas as pd
import re
import preprocessing


def personality_data_to_data_frame(path):
    """
    Return a data frame with a row for every subject
    :param path: path to txt file with personality questionnaire results
    :return: a data frame
    """
    f = open(path, 'r')
    rows = [re.split("\t+| +", line) for line in f.readlines() if 'subject' in line]
    for i in range(len(rows)):
        rows[i] = rows[i][:11]
    rows = calculate_big5_scores(rows)
    df = pd.DataFrame(rows, columns=get_column_names_traits())
    return normalize_perso_df(df)


def calculate_big5_scores(rows):
    rows_with_scores = []
    for row in rows:
        new_row = [row[0].replace('_', ''),
                   5 - int(row[1]) + int(row[6]),
                   5 - int(row[2]) + int(row[7]),
                   5 - int(row[3]) + int(row[8]),
                   5 - int(row[4]) + int(row[9]),
                   5 - int(row[5]) + int(row[10])]
        rows_with_scores.append(new_row)
    return rows_with_scores


def get_column_names_traits():
    columns = ['subject']
    columns.extend(['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience'])
    return columns


def get_column_names():
    columns = ['subject']
    columns.extend(['question_' + str(x) for x in range(1, 11)])
    columns.extend(['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience'])
    return columns


def normalize_perso_df(df):
    subject_column = df['subject']
    df = df.drop(['subject'], axis=1)
    personality_df = df.transpose()
    personality_df = preprocessing.normalize_data(personality_df, columns=personality_df.columns).transpose()
    personality_df.columns = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism',
                              'openness_to_experience']
    return pd.concat([subject_column, personality_df], axis=1)


def main():
    path = "../data/behavior_AND_personality_dataset/big5_personality_result.txt"
    path2 = "../data/data_recordings_master/personality.txt"
    # df2 = personality_data_to_data_frame(path2)
    # print df2
    # df = personality_data_to_data_frame(path)
    # print df
    # print pd.concat([df, df2], ignore_index=True)


if __name__ == "__main__":
    main()
