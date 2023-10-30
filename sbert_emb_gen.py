#TODO: one time sBERT item_feat embedding and save
from sentence_transformers import SentenceTransformer
import torch
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def get_item_feat_sbert(items_df, save_path='dataset/sbert_embeddings_full.npy'):

    def preprocess_text(text):
        return text

    preprocessed_review_texts = [preprocess_text(review_text[:512]) for review_text in df['review_text']]

    # Use SentenceTransformer to obtain embeddings
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    item_feat = model.encode(preprocessed_review_texts)

    # Save embeddings to a file
    np.save(save_path, item_feat)
    print(f"SBERT embeddings saved to {save_path}.")
    
    return item_feat

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
get_item_feat_sbert(df)