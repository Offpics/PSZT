import csv
import numpy as np
import re


def refactor(csv_path, new_csv_path):
    """ Extract sentiment and text into new csv file.

    Args:
        csv_path: File path to the csv file.
        new_csv_path: File path to the new csv file.
    """

    # Open the original csv file.
    with open(csv_path, encoding="ISO-8859-1") as csv_file:
        # Create reader object used to iterate over every row in the file.
        csv_content = csv.reader(csv_file, delimiter=',')

        # Create/replace and write new csv file.
        with open(new_csv_path, mode='w') as new_csv_path:
            # Create writer object used to write rows in new csv file.
            csv_writer = csv.writer(new_csv_path, delimiter=',')

            for row in csv_content:
                # Write every 'sentiment' and 'text' to new row.
                if row[5] != 'not_relevant':
                    csv_writer.writerow([row[5], row[11]])

def create_onehot(labels):
    one_hot = np.zeros(3)
    if int(labels) == 1:
        one_hot[0] = 1
    elif int(labels) == 3:
        one_hot[1] = 1
    elif int(labels) == 5:
        one_hot[2] = 1

    return one_hot


def vectorize_dataset(csv_path, x_train_path, y_train_path, ignore_list=None):
    """ Vectorize dataset using bag-of-words algorithm and one-hot encoding
    
    Args:
        x_train_path: Path to file to save x_train array.
        y_train_path: Path to file to save y_train array.
    """
    # List of words in the dataset.
    dataset_words = []

    # List of sentences in the dataset.
    dataset_sentences = []

    # List of sentiments in the dataset.
    dataset_sentiments = []

    # count = 0
    # Open csv dataset.
    with open(csv_path) as csv_file:
        # Create reader object used to iterate over every row in the file.
        csv_content = csv.reader(csv_file, delimiter=',')

        # Helper variable to skip first row which contains labels of csv.
        first_row = True

        # List of words to be skipped.
        if ignore_list is None:
            ignore_words = []

        for row in csv_content:

            # Skip first row of csv file.
            if first_row == True:
                first_row = False
                continue

            
            # Split every word into a list of strings.
            words = re.sub("[^\w]", " ",  row[1]).split()

            # Clean words.
            words_cleaned = [w.lower() for w in words if w.lower() not in ignore_words]

            # Change sentiments to one-hot vector.
            one_hot = create_onehot(row[0])

            dataset_words.extend(words_cleaned)
            dataset_sentences.append(words_cleaned)
            dataset_sentiments.append(one_hot)

    # Transform into set to avoid duplicates and create sorted list.
    dataset_words = sorted(list(set(dataset_words)))
    # print(dataset_words)
    
    print(len(dataset_sentences))
    print(len(dataset_words))
    # print(count)

    len_dataset_words = len(dataset_words)

    # Perform bag of words on the dataset.
    for i, sentence in enumerate(dataset_sentences):
        # Numpy array that will represent every sentence with frequencies
        # of used words.
        bag = np.zeros(len_dataset_words)

        for word_s in sentence:
            for j, word_w in enumerate(dataset_words):
                if word_w == word_s:
                    bag[j] += 1
        
        # bag = np.reshape(bag, (len_dataset_words, 1))
        dataset_sentences[i] = bag

    # Transform dataset_sentences list into numpy array.
    x_train = np.array(dataset_sentences)
    np.save(x_train_path, x_train)

    y_train = np.array(dataset_sentiments)
    np.save(y_train_path, y_train)
    return x_train, y_train


if __name__ == "__main__":
    # File path to the csv file.
    csv_path = '../Apple-Twitter-Sentiment-DFE.csv'

    # File path to the new csv file.
    new_csv_path = '../apple-twitter_test.csv'

    # Refactor dataset.
    refactor(csv_path, new_csv_path)

    x_train, y_train = vectorize_dataset(new_csv_path,
                                         'x_train_test',
                                         'y_train_test')

    # x_train = np.load('x_train_test.npy')
    # y_train = np.load('y_train_test.npy')
    # print(x_train.shape)
    # print(y_train.shape)

    # print(y_train[1])

    # for i in range(10):
    #     print(f'Sentiment: {y_train[i]}, Text: {x_train[i]}')
