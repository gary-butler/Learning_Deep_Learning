import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

data_pandas = pd.read_pickle("num_data.pkl")

plt.scatter(data_pandas.values[1:,4], data_pandas.values[1:,5])

plt.xlabel('<-negitive | positive ->')
plt.ylabel('<- emotional | subjective ->')
plt.title('Sentiment Analysis')
plt.tick_params(bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
plt.show()
