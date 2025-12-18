import pandas as pd
import numpy as np
import os


def diagnose():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    author_data = pd.read_csv('data/authorData.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"AuthorData shape: {author_data.shape}")
    train_authors = set(train['author'].unique())
    test_authors = set(test['author'].unique())
    known_authors = set(author_data['author'].unique())
    train_covered = len(train_authors.intersection(known_authors)) / len(train_authors)
    test_covered = len(test_authors.intersection(known_authors)) / len(test_authors)
    print(f"Train authors in AuthorData: {train_covered:.2%}")
    print(f"Test authors in AuthorData: {test_covered:.2%}")
    merged = train.merge(author_data[['author', 'engagement']], on='author', how='left', suffixes=('', '_avg'))
    corr = merged['engagement'].corr(merged['engagement_avg'])
    print(f"Correlation between Engagement and Author Avg Engagement: {corr:.4f}")
    train_agg = train.groupby('author')['engagement'].mean().reset_index()
    comparison = train_agg.merge(author_data, on='author', suffixes=('_train', '_data'))
    diff = (comparison['engagement_train'] - comparison['engagement_data']).abs().mean()
    print(f"Mean absolute difference between Train Agg and AuthorData: {diff:.4f}")
    if diff < 0.001:
        print("WARNING: AuthorData seems to be exactly the aggregation of Train data.")
        print("This causes LEAKAGE in validation if not handled correctly.")
    else:
        print("AuthorData seems to be independent or contain more data.")


if __name__ == "__main__":
    diagnose()
