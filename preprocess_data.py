import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

def remove_stop_words(text):
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in text.split() if word not in stop_words])

def prepare_blog_dataset(dataset_path='dataset'):
    print("Getting Blog dataset...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rtatman/blog-authorship-corpus",
        "blogtext.csv",
    )

    print("Processing Blog dataset...")
    # Calculate the number of words in each blog post
    text_len = df.text.apply(lambda x: len(RegexpTokenizer(r'\w+').tokenize(x))).to_numpy()

    # Combine the text length with the dataset
    df['text_len'] = text_len

    # Remove posts with less than 100 words
    df = df[df['text_len'] > 100]

    # Get the number of posts for each author
    authors = df['id'].to_numpy()
    author_id, counts = np.unique(authors, return_counts=True)

    # Remove authors with less than 10 posts
    valid_authors = author_id[counts > 10]
    df = df[df['id'].isin(valid_authors)]

    # Strip text
    df['text'] = df['text'].str.strip()

    punctuation = ['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', "'", '"']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    short_word_len = 4

    # We calculate the stylometric features
    word_len = np.zeros(len(df))
    sentence_len = np.zeros(len(df))
    short_words = np.zeros(len(df))
    digit_prop = np.zeros(len(df))
    captialized_prop = np.zeros(len(df))
    letter_freq = np.zeros((len(df), len(letters)))
    digit_freq = np.zeros((len(df), len(digits)))
    punctuation_freq = np.zeros((len(df), len(punctuation)))
    hapax_legomena = np.zeros(len(df))

    for i in tqdm(range(len(df))):
        text = df['text'].iloc[i]
        sentences = nltk.sent_tokenize(text)
        words = RegexpTokenizer(r'\w+').tokenize(text)

        word_lengths = [len(word) for word in words]

        if len(word_lengths) == 0:
            print(f"Empty word length in post {i}")
            break

        if len([len(sentence.split(' ')) for sentence in sentences]) == 0:
            print(f"Empty sentence in post {i}")
            break

        word_len[i] = np.mean(word_lengths)
        short_words[i] = np.sum([1 for word in words if len(word) < short_word_len])
        sentence_len[i] = np.mean([len(sentence.split(' ')) for sentence in sentences])

        character_counts = Counter(list(text.lower()))

        for j, letter in enumerate(letters):
            if letter in character_counts:
                letter_freq[i][j] = character_counts[letter]
            else:
                letter_freq[i][j] = 0

        for j, digit in enumerate(digits):
            if digit in character_counts:
                digit_freq[i][j] = character_counts[digit]
            else:
                digit_freq[i][j] = 0

        for j, punct in enumerate(punctuation):
            if punct in character_counts:
                punctuation_freq[i][j] = character_counts[punct]
            else:
                punctuation_freq[i][j] = 0

        letter_freq[i] /= np.sum(letter_freq[i]) + 1e-10
        digit_freq[i] /= np.sum(digit_freq[i]) + 1e-10
        punctuation_freq[i] /= np.sum(punctuation_freq[i]) + 1e-10
        
        hapax_legomena[i] = len([word for word, count in Counter(words).items() if count == 1])

        text_len = df['text_len'].iloc[i]

        digit_prop[i] = np.sum([1 for word in words if word.isdigit()]) / text_len
        captialized_prop[i] = np.sum([1 for word in words if word[0].isupper()]) / text_len

    # mean_word_len = np.mean(word_len)
    # mean_sentence_len = np.mean(sentence_len)
    # mean_short_words = np.mean(short_words)
    # mean_hapax_legomena = np.mean(hapax_legomena)

    # std_word_len = np.std(word_len)
    # std_sentence_len = np.std(sentence_len)
    # std_short_words = np.std(short_words)
    # std_hapax_legomena = np.std(hapax_legomena)

    # # Normalize the features
    # word_len = (word_len - mean_word_len) / std_word_len
    # sentence_len = (sentence_len - mean_sentence_len) / std_sentence_len
    # short_words = (short_words - mean_short_words) / std_short_words
    # hapax_legomena = (hapax_legomena - mean_hapax_legomena) / std_hapax_legomena

    # Add the features to the dataframe
    df['word_len'] = word_len
    df['sentence_len'] = sentence_len
    df['short_words'] = short_words
    df['digit_prop'] = digit_prop
    df['captialized_prop'] = captialized_prop
    df['hapax_legomena'] = hapax_legomena

    # Add the letter frequency features to the dataframe with the column names
    for i in range(len(letters)):
        df[f'letter_freq_{letters[i]}'] = letter_freq[:, i]
    for i in range(len(digits)):
        df[f'digit_freq_{digits[i]}'] = digit_freq[:, i]
    for i in range(len(punctuation)):
        df[f'punctuation_freq_{punctuation[i]}'] = punctuation_freq[:, i]

    # # Remove stop words
    df['text'] = df['text'].apply(remove_stop_words)

    # Drop the features that are not needed
    df = df.drop(columns=['gender', 'age', 'topic', 'sign', 'date'])

    return df

def save_blog_dataset(df, dataset_path='dataset'):
    # Split the dataset into training and testing sets by author
    authors = df['id'].unique()

    np.random.seed(42)
    np.random.shuffle(authors)

    train_size = 0.8
    train_authors = authors[:int(len(authors) * train_size)]
    test_authors = authors[int(len(authors) * train_size):]

    train_df = df[df['id'].isin(train_authors)]
    test_df = df[df['id'].isin(test_authors)]

    # Save the training and testing sets
    train_df.to_csv(os.path.join(dataset_path, 'blogtext_train_all.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_path, 'blogtext_test_all.csv'), index=False)

    # Get top 10 authors with the most posts
    top_authors = df['id'].value_counts().nlargest(10).index
    # Only keep the posts from the top 10 authors
    df_10 = df[df['id'].isin(top_authors)]
    # Create training and testing sets for the top 10 authors
    # Shuffle the dataframe
    df_10 = df_10.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset into training and testing sets
    train_size = 0.8
    train_df_10 = df_10[:int(len(df_10) * train_size)]
    test_df_10 = df_10[int(len(df_10) * train_size):]
    # Save the training and testing sets
    train_df_10.to_csv(os.path.join(dataset_path, 'blogtext_train_10.csv'), index=False)
    test_df_10.to_csv(os.path.join(dataset_path, 'blogtext_test_10.csv'), index=False)

    # Get top 50 authors with the most posts
    top_authors = df['id'].value_counts().nlargest(50).index

    # Only keep the posts from the top 50 authors
    df_50 = df[df['id'].isin(top_authors)]

    # Create training and testing sets for the top 50 authors
    # Shuffle the dataframe
    df_50 = df_50.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset into training and testing sets
    train_size = 0.8
    train_df_50 = df_50[:int(len(df_50) * train_size)]
    test_df_50 = df_50[int(len(df_50) * train_size):]

    # Save the training and testing sets
    train_df_50.to_csv(os.path.join(dataset_path, 'blogtext_train_50.csv'), index=False)
    test_df_50.to_csv(os.path.join(dataset_path, 'blogtext_test_50.csv'), index=False)

def prepare_imdb62_dataset(dataset_path='dataset'):
    print("Getting IMDB62 dataset...")
    df = pd.read_parquet("hf://datasets/tasksource/imdb62/data/train-00000-of-00001-62894f3b39974716.parquet")

    print("Processing IMDB62 dataset...")

    # Calculate the number of words in each review
    text_len = df.content.apply(lambda x: len(RegexpTokenizer(r'\w+').tokenize(str(x)))).to_numpy()
    # Combine the text length with the dataset
    df['text_len'] = text_len
    # Remove reviews with less than 100 words
    df = df[df['text_len'] > 100]
    # Strip text
    df['content'] = df['content'].str.strip()

    punctuation = ['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', "'", '"']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    short_word_len = 4

    # We calculate the stylometric features
    word_len = np.zeros(len(df))
    sentence_len = np.zeros(len(df))
    short_words = np.zeros(len(df))
    digit_prop = np.zeros(len(df))
    captialized_prop = np.zeros(len(df))
    letter_freq = np.zeros((len(df), len(letters)))
    digit_freq = np.zeros((len(df), len(digits)))
    punctuation_freq = np.zeros((len(df), len(punctuation)))
    hapax_legomena = np.zeros(len(df))

    for i in tqdm(range(len(df))):
        text = df['content'].iloc[i]
        sentences = nltk.sent_tokenize(text)
        words = RegexpTokenizer(r'\w+').tokenize(text)

        word_lengths = [len(word) for word in words]

        if len(word_lengths) == 0:
            print(f"Empty word length in post {i}")
            break

        if len([len(sentence.split(' ')) for sentence in sentences]) == 0:
            print(f"Empty sentence in post {i}")
            break

        word_len[i] = np.mean(word_lengths)
        short_words[i] = np.sum([1 for word in words if len(word) < short_word_len])
        sentence_len[i] = np.mean([len(sentence.split(' ')) for sentence in sentences])

        character_counts = Counter(list(text.lower()))

        for j, letter in enumerate(letters):
            if letter in character_counts:
                letter_freq[i][j] = character_counts[letter]
            else:
                letter_freq[i][j] = 0

        for j, digit in enumerate(digits):
            if digit in character_counts:
                digit_freq[i][j] = character_counts[digit]
            else:
                digit_freq[i][j] = 0

        for j, punct in enumerate(punctuation):
            if punct in character_counts:
                punctuation_freq[i][j] = character_counts[punct]
            else:
                punctuation_freq[i][j] = 0

        letter_freq[i] /= np.sum(letter_freq[i]) + 1e-10
        digit_freq[i] /= np.sum(digit_freq[i]) + 1e-10
        punctuation_freq[i] /= np.sum(punctuation_freq[i]) + 1e-10
        
        hapax_legomena[i] = len([word for word, count in Counter(words).items() if count == 1])

        text_len = df['text_len'].iloc[i]

        digit_prop[i] = np.sum([1 for word in words if word.isdigit()]) / text_len
        captialized_prop[i] = np.sum([1 for word in words if word[0].isupper()]) / text_len

    # Add the features to the dataframe
    df['word_len'] = word_len
    df['sentence_len'] = sentence_len
    df['short_words'] = short_words
    df['digit_prop'] = digit_prop
    df['captialized_prop'] = captialized_prop
    df['hapax_legomena'] = hapax_legomena

    # Add the letter frequency features to the dataframe with the column names
    for i in range(len(letters)):
        df[f'letter_freq_{letters[i]}'] = letter_freq[:, i]
    for i in range(len(digits)):
        df[f'digit_freq_{digits[i]}'] = digit_freq[:, i]
    for i in range(len(punctuation)):
        df[f'punctuation_freq_{punctuation[i]}'] = punctuation_freq[:, i]

    # Remove stop words
    df['content'] = df['content'].apply(remove_stop_words)

    # Drop the features that are not needed
    df = df.drop(columns=['reviewId', 'itemId', 'rating', 'title'])

    return df

def save_imdb62_dataset(df, dataset_path='dataset'):

    # Split the dataset into training and testing sets by author
    authors = df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(authors)
    train_size = 0.8
    train_authors = authors[:int(len(authors) * train_size)]
    test_authors = authors[int(len(authors) * train_size):]
    train_df = df[df['id'].isin(train_authors)]
    test_df = df[df['id'].isin(test_authors)]
    # Save the training and testing sets
    train_df.to_csv(os.path.join(dataset_path, 'imdb62_train.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_path, 'imdb62_test.csv'), index=False)

    # Split randomly
    df_62 = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset into training and testing sets
    train_size = 0.8
    train_df_62 = df_62[:int(len(df_62) * train_size)]
    test_df_62 = df_62[int(len(df_62) * train_size):]

    # Save the training and testing sets
    train_df_62.to_csv(os.path.join(dataset_path, 'imdb62_train_62.csv'), index=False)
    test_df_62.to_csv(os.path.join(dataset_path, 'imdb62_test_62.csv'), index=False)

def prepare_imdb_dataset(dataset_path='dataset'):
    print("Getting IMDB1M dataset...")
    df_reviews = pd.read_csv("imdb1m-reviews.txt", sep="\t", names=["reviewId", "id", "itemId", "rating", "title", "text"], header=None)
    df_posts = pd.read_csv("imdb1m-posts.txt", sep="\t", names=["reviewId", "id", "title", "text"], header=None)
    df_reviews = df_reviews.drop(columns=["reviewId", "itemId", "rating", "title"])
    df_posts = df_posts.drop(columns=["reviewId", "title"])
    df = pd.concat([df_reviews, df_posts], ignore_index=True)

    print("Processing IMDB1M dataset...")

    # Calculate the number of words in each review
    text_len = df.text.apply(lambda x: len(RegexpTokenizer(r'\w+').tokenize(str(x)))).to_numpy()
    # Combine the text length with the dataset
    df['text_len'] = text_len
    # Remove reviews with less than 100 words
    df = df[df['text_len'] > 100]
    # Strip text
    df['text'] = df['text'].str.strip()

    # Get the number of posts for each author
    authors = df['id'].to_numpy()
    author_id, counts = np.unique(authors, return_counts=True)

    # Remove authors with less than 10 posts
    valid_authors = author_id[counts > 10]
    df = df[df['id'].isin(valid_authors)]

    punctuation = ['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', "'", '"']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    short_word_len = 4

    # We calculate the stylometric features
    word_len = np.zeros(len(df))
    sentence_len = np.zeros(len(df))
    short_words = np.zeros(len(df))
    digit_prop = np.zeros(len(df))
    captialized_prop = np.zeros(len(df))
    letter_freq = np.zeros((len(df), len(letters)))
    digit_freq = np.zeros((len(df), len(digits)))
    punctuation_freq = np.zeros((len(df), len(punctuation)))
    hapax_legomena = np.zeros(len(df))

    for i in tqdm(range(len(df))):
        text = df['text'].iloc[i]
        sentences = nltk.sent_tokenize(text)
        words = RegexpTokenizer(r'\w+').tokenize(text)

        word_lengths = [len(word) for word in words]

        if len(word_lengths) == 0:
            print(f"Empty word length in post {i}")
            break

        if len([len(sentence.split(' ')) for sentence in sentences]) == 0:
            print(f"Empty sentence in post {i}")
            break

        word_len[i] = np.mean(word_lengths)
        short_words[i] = np.sum([1 for word in words if len(word) < short_word_len])
        sentence_len[i] = np.mean([len(sentence.split(' ')) for sentence in sentences])

        character_counts = Counter(list(text.lower()))

        for j, letter in enumerate(letters):
            if letter in character_counts:
                letter_freq[i][j] = character_counts[letter]
            else:
                letter_freq[i][j] = 0

        for j, digit in enumerate(digits):
            if digit in character_counts:
                digit_freq[i][j] = character_counts[digit]
            else:
                digit_freq[i][j] = 0

        for j, punct in enumerate(punctuation):
            if punct in character_counts:
                punctuation_freq[i][j] = character_counts[punct]
            else:
                punctuation_freq[i][j] = 0

        letter_freq[i] /= np.sum(letter_freq[i]) + 1e-10
        digit_freq[i] /= np.sum(digit_freq[i]) + 1e-10
        punctuation_freq[i] /= np.sum(punctuation_freq[i]) + 1e-10
        
        hapax_legomena[i] = len([word for word, count in Counter(words).items() if count == 1])

        text_len = df['text_len'].iloc[i]

        digit_prop[i] = np.sum([1 for word in words if word.isdigit()]) / text_len
        captialized_prop[i] = np.sum([1 for word in words if word[0].isupper()]) / text_len

    # Add the features to the dataframe
    df['word_len'] = word_len
    df['sentence_len'] = sentence_len
    df['short_words'] = short_words
    df['digit_prop'] = digit_prop
    df['captialized_prop'] = captialized_prop
    df['hapax_legomena'] = hapax_legomena

    # Add the letter frequency features to the dataframe with the column names
    for i in range(len(letters)):
        df[f'letter_freq_{letters[i]}'] = letter_freq[:, i]
    for i in range(len(digits)):
        df[f'digit_freq_{digits[i]}'] = digit_freq[:, i]
    for i in range(len(punctuation)):
        df[f'punctuation_freq_{punctuation[i]}'] = punctuation_freq[:, i]

    # Remove stop words
    df['text'] = df['text'].apply(remove_stop_words)

    return df

def save_imdb_dataset(df, dataset_path='dataset'):
    # Split the dataset into training and testing sets by author
    authors = df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(authors)
    train_size = 0.8
    train_authors = authors[:int(len(authors) * train_size)]
    test_authors = authors[int(len(authors) * train_size):]
    train_df = df[df['id'].isin(train_authors)]
    test_df = df[df['id'].isin(test_authors)]
    # Save the training and testing sets
    train_df.to_csv(os.path.join(dataset_path, 'imdb_train.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_path, 'imdb_test.csv'), index=False)

    # Get top 100 authors with the most posts
    top_authors = df['id'].value_counts().nlargest(100).index
    # Only keep the posts from the top 100 authors
    df_100 = df[df['id'].isin(top_authors)]
    # Create training and testing sets for the top 100 authors
    # Shuffle the dataframe
    df_100 = df_100.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset into training and testing sets
    train_size = 0.8
    train_df_100 = df_100[:int(len(df_100) * train_size)]
    test_df_100 = df_100[int(len(df_100) * train_size):]
    # Save the training and testing sets
    train_df_100.to_csv(os.path.join(dataset_path, 'imdb_train_100.csv'), index=False)
    test_df_100.to_csv(os.path.join(dataset_path, 'imdb_test_100.csv'), index=False)

def main():

    nltk.download('punkt_tab')
    nltk.download('stopwords')

    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    df_blog = prepare_blog_dataset(dataset_path)
    
    df_imdb62 = prepare_imdb62_dataset(dataset_path)

    df_imdb = prepare_imdb_dataset(dataset_path)

    # Rename the userId column to id in the imdb dataset
    df_imdb62.rename(columns={'userId': 'id'}, inplace=True)
    # Rename the content column to text in the imdb dataset
    df_imdb62.rename(columns={'content': 'text'}, inplace=True)

    # Calculate mean and std of the stylometric features in the blog dataset
    mean_word_len = np.mean(df_blog['word_len'])
    mean_sentence_len = np.mean(df_blog['sentence_len'])
    mean_short_words = np.mean(df_blog['short_words'])
    mean_hapax_legomena = np.mean(df_blog['hapax_legomena'])
    std_word_len = np.std(df_blog['word_len'])
    std_sentence_len = np.std(df_blog['sentence_len'])
    std_short_words = np.std(df_blog['short_words'])
    std_hapax_legomena = np.std(df_blog['hapax_legomena'])

    # Normalize the features in the blog dataset
    df_blog['word_len'] = (df_blog['word_len'] - mean_word_len) / std_word_len
    df_blog['sentence_len'] = (df_blog['sentence_len'] - mean_sentence_len) / std_sentence_len
    df_blog['short_words'] = (df_blog['short_words'] - mean_short_words) / std_short_words
    df_blog['hapax_legomena'] = (df_blog['hapax_legomena'] - mean_hapax_legomena) / std_hapax_legomena
    save_blog_dataset(df_blog, dataset_path)

    # Normalise the features in the imdb62 dataset using the blog dataset statistics
    df_imdb62['word_len'] = (df_imdb62['word_len'] - mean_word_len) / std_word_len
    df_imdb62['sentence_len'] = (df_imdb62['sentence_len'] - mean_sentence_len) / std_sentence_len
    df_imdb62['short_words'] = (df_imdb62['short_words'] - mean_short_words) / std_short_words
    df_imdb62['hapax_legomena'] = (df_imdb62['hapax_legomena'] - mean_hapax_legomena) / std_hapax_legomena
    save_imdb62_dataset(df_imdb62, dataset_path)

    # Normalize the features in the imdb dataset using the blog dataset statistics
    df_imdb['word_len'] = (df_imdb['word_len'] - mean_word_len) / std_word_len
    df_imdb['sentence_len'] = (df_imdb['sentence_len'] - mean_sentence_len) / std_sentence_len
    df_imdb['short_words'] = (df_imdb['short_words'] - mean_short_words) / std_short_words
    df_imdb['hapax_legomena'] = (df_imdb['hapax_legomena'] - mean_hapax_legomena) / std_hapax_legomena
    save_imdb_dataset(df_imdb, dataset_path)

    # Reassign the author ids to be consecutive integers starting from 0
    blog_author_ids = df_blog['id'].unique()
    blog_author_id_map = {author_id: i for i, author_id in enumerate(blog_author_ids)}
    df_blog['id'] = df_blog['id'].map(blog_author_id_map)

    imdb_author_ids = df_imdb['id'].unique()
    imdb_author_id_map = {author_id: i + len(blog_author_ids) for i, author_id in enumerate(imdb_author_ids)}
    df_imdb['id'] = df_imdb['id'].map(imdb_author_id_map)
    
    # Combine the datasets
    combined_df = pd.concat([df_blog, df_imdb], ignore_index=True)

    # Split the combined dataset into training and testing sets
    authors = combined_df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(authors)
    train_size = 0.8
    train_authors = authors[:int(len(authors) * train_size)]
    test_authors = authors[int(len(authors) * train_size):]
    train_df = combined_df[combined_df['id'].isin(train_authors)]
    test_df = combined_df[combined_df['id'].isin(test_authors)]
    # Save the training and testing sets
    train_df.to_csv(os.path.join(dataset_path, 'combined_train_all.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_path, 'combined_test_all.csv'), index=False)
    print("Datasets prepared and saved successfully.")


if __name__ == "__main__":
    main()