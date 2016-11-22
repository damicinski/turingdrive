# import csv
# import os

import pandas as pd

# from os import listdir
# from os.path import splitext
# from pandas import DataFrame
#
# DIR_PATH = './center'
#
# timestamps = [splitext(f)[0] for f in listdir(DIR_PATH) if f.endswith('.png')]
# print type(image_file_names)
# print image_file_names

csv_file = 'interpolated.csv'
df = pd.read_csv(csv_file)

# print df.head()
# df_selected = df[df.frame_id == 'center_camera']['frame_id', 'timestamp']

df_selected = df.loc[df.frame_id=='center_camera',
                     ['frame_id', 'timestamp', 'angle']]

df_selected.rename(columns = {'frame_id':'camera',
                              'timestamp':'frame_id',
                              'angle':'steering_angle'}, inplace=True)

df_selected.drop(['camera'], axis=1, inplace=True)

# print df_selected.head()

df_selected.to_csv('center.csv', index=False)
# saved_column
#
# with open('interpolated.csv', 'rb') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print row
