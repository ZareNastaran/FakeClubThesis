from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import torch
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

embedding_size = 100

def get_item_feat_word2vec(items_df, save_path='dataset/word2vec_embeddings_full.npy'):
    sentences = df['review_text'].apply(lambda x: word_tokenize(x.lower()))
    model = Word2Vec(sentences=sentences, vector_size=embedding_size, window=5, min_count=1, workers=4)
    model.train(sentences, total_examples=len(sentences), epochs=50)

    # Average the word vectors for each sentence for our dataset
    def get_avg_vector(words, model, num_features):
        feature_vector = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        vocabulary = set(model.wv.index_to_key)

        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    edge_feat = np.array(sentences.apply(lambda x: get_avg_vector(x, model, embedding_size)).tolist())

    # Save embeddings to a file
    np.save(save_path, edge_feat)
    print(f"Word2vec embeddings saved to {save_path}.")

    return edge_feat

df = pd.read_parquet('dataset/labeled_yelp_dataset_ver2_small.parquet')
def encode_id(df):
    business_encoder = LabelEncoder()
    df['business_id'] = business_encoder.fit_transform(df['business_id'])
    user_encoder = LabelEncoder()
    mapping = dict(zip(df['user_id'] ,user_encoder.fit_transform(df['user_id'])))
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    def map_friends(friends_str, mapping_dict):
        return [mapping_dict[friend] for friend in friends_str.split(',') if friend in mapping_dict]

    df['friends'] = df['friends'].apply(map_friends, mapping_dict=mapping)
    return df

df = encode_id(df)
df = df[['user_id', 'friends', 'business_id', 'stars', 'text', 'label', 'name']]
df = df.rename(columns={
    'user_id': 'user',
    'business_id': 'item',
    'stars': 'rating',
    'text': 'review_text',
    'name': 'business_name',
})
df = df.drop_duplicates(subset=['user', 'item'], keep='first')
get_item_feat_word2vec(df)