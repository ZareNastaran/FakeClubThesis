from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

Yelp_df = pd.read_parquet("dataset/yelp_dataset_ver1.parquet")

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(Yelp_df['Tokens'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
Yelp_data = pd.concat([Yelp_df, tfidf_df], axis=1)

Yelp_data.to_parquet("dataset/yelp_dataset_with_tfidf_features.parquet")