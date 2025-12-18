import pandas as pd


def check_overlap():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train_authors = set(train['author'].unique())
    test_authors = set(test['author'].unique())
    print(f"Unique authors in Train: {len(train_authors)}")
    print(f"Unique authors in Test: {len(test_authors)}")
    common_authors = train_authors.intersection(test_authors)
    print(f"Authors in both Train and Test: {len(common_authors)}")
    pct_test_in_train = len(common_authors) / len(test_authors) * 100
    print(f"% of Test authors seen in Train: {pct_test_in_train:.2f}%")
    new_authors = test_authors - train_authors
    print(f"New authors in Test (Cold Start): {len(new_authors)}")


if __name__ == "__main__":
    check_overlap()
