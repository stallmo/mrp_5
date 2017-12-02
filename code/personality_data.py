import pandas as pd
import re


def personality_data_to_data_frame(path):
    """
    Return a data frame with a row for every subject
    :param path: path to txt file with personality questionnaire results
    :return: a data frame
    """
    file = open(path, 'r')
    rows = [re.split("\t+", line) for line in file.readlines() if 'subject' in line]
    rows = clean_data(rows)
    df = pd.DataFrame(rows, columns=get_column_names())
    return df


def get_column_names():
    columns = ['name']
    columns.extend(['question_' + str(x) for x in range(1, 11)])
    columns.extend(['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience'])
    return columns


def clean_data(list_2d):
    cleaned_list = []
    for row in list_2d:
        for index, value in enumerate(row):
            if ':' in value:
                row[index] = re.findall('\d+', value)[0]
        cleaned_list.append(row)
    return cleaned_list


def main():
    path = "../data/behavior_AND_personality_dataset/big5_personality_result.txt"
    print(personality_data_to_data_frame(path))


if __name__ == "__main__":
    main()
