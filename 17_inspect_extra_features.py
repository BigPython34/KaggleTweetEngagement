import pandas as pd

def inspect_features():
    df = pd.read_csv('data/train.csv', nrows=1000)
    print("Columns:", df.columns.tolist())
    print("\nFeature1 stats:")
    print(df['feature1'].describe())
    print(df['feature1'].value_counts().head())
    
    print("\nFeature2 stats:")
    print(df['feature2'].describe())
    print(df['feature2'].value_counts().head())
    
    print("\nLanguage stats:")
    print(df['language'].value_counts().head())
    
    # Check if feature1/2 are constant per author
    author_feat = df.groupby('author')[['feature1', 'feature2']].nunique()
    print("\nMax unique feature values per author (if 1, it's an author property):")
    print(author_feat.max())

if __name__ == "__main__":
    inspect_features()
