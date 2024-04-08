#                                                     In The Name of GOD                                                    #
##                                                           Be Aware                                                      ##
##                                                           Be Aware                                                      ##
##                                                                                                                         ##
##                                      %  DO NOT Replace OR Remove The 'stop_words' Folder  %                             ##
##                                                                                                                         ##
##                                          Unless You are Fully Aware of The Consequences                                 ##
##                                                                                                                         ##
##                                                           Be Aware                                                      ##
##                                                           Be Aware                                                      ##
# https://github.com/cloner174
# cloner174.org@gmail.com
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hazm import word_tokenize
from stop_words import get_stop_words
import time

# Load data
file_path = 'data/SimilarityCorpusSamples.xlsx'
try:
  df = pd.read_excel(file_path)
except:
  print(" Please put your data in the data folder inside ->main directory ")
  time.sleep(1)
  print(" Please rename your data to be like this-->> SimilarityCorpusSamples.xlsx ")
  time.sleep(2)
  print(" Otherwise there is nothing I can do ")
  time.sleep(1)
  print(f" Closing the Program ....")
  exit()

try:
  df.columns = ["sentence1", "sentence2", "score"]
except:
  raise ValueError("\n   You should have a  data  with 3 columns -->> \sentence1\, \sentence2\, \score\ <<--  \n")
  exit()


# Normalize the data and remove stop words
stop_words = get_stop_words('persian')
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['sentence1'] + ' ' + df['sentence2'])
df_normalized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_normalized['Similarity'] = df['score']

# Save normalized data to a new file
TemP = np.random.randint(10000)
normalized_file_path = f'Output/normalized_corpus_testNO{TemP}.xlsx'
df_normalized.to_excel(normalized_file_path, index=False)

# Extract the corpus vocabulary and save it to Dic.txt
corpus_vocab = vectorizer.get_feature_names_out()
with open('Dic.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(corpus_vocab))

# Calculate similarity using TF and cosine similarity
def calculate_similarity_TF(s1, s2):
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tf_matrix = vectorizer.fit_transform([s1, s2])
    similarity = cosine_similarity(tf_matrix[0], tf_matrix[1])[0, 0]
    return similarity

# Calculate Jaccard Similarity
def jaccard_similarity(s1, s2):
    tokens_s1 = set(word_tokenize(s1))
    tokens_s2 = set(word_tokenize(s2))
    intersection = len(tokens_s1.intersection(tokens_s2))
    union = len(tokens_s1.union(tokens_s2))
    return intersection / union if union != 0 else 0

# Calculate similarities and create result DataFrame
result_df = {
    "sentence1": df['sentence1'],
    "sentence2": df['sentence2'],
    "Real Score": df['score'],
    "Similarity TF": [calculate_similarity_TF(s1, s2) for s1, s2 in zip(df['sentence1'], df['sentence2'])],
    "Jaccard Similarity": [jaccard_similarity(s1, s2) for s1, s2 in zip(df['sentence1'], df['sentence2'])]
}
result_df = pd.DataFrame(result_df)

# Calculate average similarities
average_tf_similarity = np.mean(result_df["Similarity TF"])
average_jaccard_similarity = np.mean(result_df["Jaccard Similarity"])
real_average = np.mean(df['score'])

# Print results
print("\n Real Average Score: {:.5f}".format(real_average))
time.sleep(2)
print("\n Average TF Cosine Similarity: {:.5f}".format(average_tf_similarity))
time.sleep(3)
print("\n Average Jaccard Similarity: {:.5f}".format(average_jaccard_similarity))
time.sleep(3)
print("\n All the files that is generated from this scripts , \n    is whether in main directory or in Output Folder " )
time.sleep(2)
#end#
