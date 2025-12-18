import pandas as pd
import os


DATA_DIR = 'data'
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
AUTHOR_DATA_PATH = os.path.join(DATA_DIR, 'authorData.csv')


def check_test_authors():
    print("Checking author coverage in Test set...")
    test = pd.read_csv(TEST_PATH)
    author_data = pd.read_csv(AUTHOR_DATA_PATH)
    test_authors = set(test['author'].unique())
    known_authors = set(author_data['author'].unique())
    unknown_authors = test_authors - known_authors
    print(f"Total authors in Test: {len(test_authors)}")
    print(f"Authors found in authorData: {len(test_authors) - len(unknown_authors)}")
    print(f"New/Unknown authors: {len(unknown_authors)}")
    print(f"Percentage of unknown authors: {len(unknown_authors) / len(test_authors) * 100:.2f}%")
    unknown_rows = test[test['author'].isin(unknown_authors)]
    print(f"Test rows with unknown authors: {len(unknown_rows)} / {len(test)} ({len(unknown_rows)/len(test)*100:.2f}%)")


if __name__ == "__main__":
    check_test_authors()
