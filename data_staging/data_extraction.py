import pandas as pd
import os
import json
import ast
from collections import ChainMap
import numpy as np
import string

CHANNEL = "msnbc"
DATA_DIR = "/Users/harshitrajgarhia/PycharmProjects/DSCI789/data/"
CHANNEL_DIR = DATA_DIR + os.sep + CHANNEL + "_parsed"
METADATA_DIR = CHANNEL_DIR + os.sep + "metadata"
JSON_BY_LINE_DIR = CHANNEL_DIR + os.sep + "jsonbyline"
VIDEO_DATA_DIR = DATA_DIR + os.sep + "DSCI-789-84740"
FILTERED_DATA_DIR = DATA_DIR+ os.sep + "filtered_data"

#filtered_comments_df = pd.read_hdf(FILTERED_DATA_DIR+os.sep+"csv_files"+os.sep+CHANNEL+'_comments.hdf',CHANNEL+'_df')
filtered_video_df = pd.read_hdf(FILTERED_DATA_DIR+os.sep+"csv_files"+os.sep+CHANNEL+'_video.hdf',CHANNEL+'_video_df')

def get_paragraphs(row):
    video_id = row['id']
    #video_id = 'CuhMGxOPxR4'
    print("******************* "+video_id)
    video_comments_list = []
    video_id_list_with_no_comments = []
    video_comments_list_text = []
    try:
        with open(JSON_BY_LINE_DIR + os.sep + video_id + "_jsonbyline.txt",'rb') as f:
            for line in f:
                print(line)
                dict_str = line.decode('utf-8')  # converting from byte class to utf-8
                dict_str = ast.literal_eval(dict_str)  # Evaluates the string can be parsed or not
                dict_str['text'] = dict_str['text'].lower()  # converts the text to lowercase
                dict_str['text'] = dict_str['text'].translate(str.maketrans('', '', string.punctuation))  # removing punctuation
                video_comments_list.append(dict_str)
                print('done')
                #video_comments_list.append(ast.literal_eval(line))
        video_comments_list_text = [comments_dict['text'] for comments_dict in video_comments_list]

    except:
        video_id_list_with_no_comments.append(video_id)

    row['paragraphs'] = video_comments_list_text
    return row


filtered_video_df_with_comments = filtered_video_df.apply(lambda x: get_paragraphs(x), axis=1)
filtered_video_df_with_comments.to_csv(FILTERED_DATA_DIR+os.sep+"qna_data"+os.sep+CHANNEL+"_video_qna_data.csv",index=False)

######################
filtered_video_df_with_comments = pd.read_csv(FILTERED_DATA_DIR+os.sep+"qna_data"+os.sep+CHANNEL+"_video_qna_data.csv")