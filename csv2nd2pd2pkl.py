import pandas as pd
from matplotlib import pyplot as plt
import csv
import numpy as np
import pickle


file = "new_all_data.csv"

np_array =['tweet_id','text','user_id','user_screen_name','user_name','created_at','retweets','likes','replies','polarity','subjectivity']

with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        np_array = np.vstack(([np_array, [row['tweet_id'],row['text'],row['user_id'],row['user_screen_name'],row['user_name'],row['created_at'],row['retweets'],row['likes'],row['replies'],row['polarity'],row['subjectivity']]]))
       
data_pandas = pd.DataFrame(np_array)

print(data_pandas)
data_pandas.to_pickle("all_data.pkl")