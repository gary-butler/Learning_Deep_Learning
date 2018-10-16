import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from os import path
from wordcloud import WordCloud, STOPWORDS
import re

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

text = open(path.join(d, "hashtag_words_edited.txt")).read()
print(text)
text = re.sub(r'http\S+', '', text)
text = re.sub(r'[a-z]*[:.]+\S+', '', text)
text = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '' ,text)

print(text)
file = open("hashtag_words_edit.txt", "w")
file.write(text)
file.close()

mask = np.array(Image.open(path.join(d, "sprite0.png")))

stopwords = set(STOPWORDS)

cloud = WordCloud(background_color="white", max_words=2000, mask=mask, stopwords=stopwords).generate(text)

cloud.to_file(path.join(d, "wcloud.png"))

plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()