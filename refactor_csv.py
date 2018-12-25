import csv
import numpy as np
import re
from itertools import chain


def refactor(csv_path, new_csv_path):
    """Extract sentiment and text into new csv file.
    Args:
    csv_path -- file path to the csv file
    new_csv_path -- file path to the new csv file
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
                csv_writer.writerow([row[5], row[11]])


def vectorize_dataset(csv_path, numpy_path):
    """Perform bag of words algorithm on the dataset.
    TODO: 
    Right now this function only performs bag-of-words on the dataset 
    and returns every sentence as vector in form of numpy array. 
    We need to add label (sentiment) to every sentence but tbh I'm not sure
    how the neural net model will look like.
    """
    # List of words in the dataset.
    dataset_words = []

    # List of sentences in the dataset.
    dataset_sentences = []

    # Open csv dataset.
    with open(csv_path) as csv_file:
        #Create reader object used to iterate over every row in the file.
        csv_content = csv.reader(csv_file, delimiter=',')

        for row in csv_content:
            #TODO: Skip first row because it has labels in it.
            ignore_words = ['a']
            # Split every words into a list.
            words = re.sub("[^\w]", " ",  row[1]).split()

            # Clean words.
            words = [w.lower() for w in words if w not in ignore_words]

            dataset_words.extend(words)
            dataset_sentences.append(words)

    # Transform into set to avoid duplicates and create sorted list.
    dataset_words = sorted(list(set(dataset_words)))

    # Perform bag of words on the dataset.
    for i, sentence in enumerate(dataset_sentences):
        # Numpy array that will represent every sentence with frequencies
        # of used words.
        bag = np.zeros(len(dataset_words))

        for word_s in sentence:
            for j, word_w in enumerate(dataset_words):
                if word_w == word_s:
                    bag[j] += 1
        
        dataset_sentences[i] = bag

    # Transform dataset_sentences list into numpy array.
    a = np.array(dataset_sentences)
    np.save(numpy_path, a)
    return a


def readWords(filename): #read words from refracted file
    file = open(filename)
    reader = csv.reader(file)
    header = next(reader)
    
    def readInput(row):
        return [par for par in row[1:]]
    
    words = [readInput(next(reader))]
    
    for row in reader:
        inp = readInput(row)
        words.append(inp)
    return [words]


def readSentiments(filename): # read sentiments from refracted file
    file = open(filename)
    reader = csv.reader(file)
    header = next(reader)

    def readSent(row):
        return row[0]

    sentiments = [readSent(next(reader))]

    for row in reader:
        sent=readSent(row)
        sentiments.append(sent)
    return [sentiments]



if __name__ == "__main__":
    #dataset = vectorize_dataset('apple-twitter.csv', 'test')
    # print(len(dataset[2]))
    # # File path to the csv file.
    # csv_path = 'Apple-Twitter-Sentiment-DFE.csv'

    # # File path to the new csv file.
    # new_csv_path = 'apple-twitter.csv'

    # # Refactor dataset.
    # refactor(csv_path, new_csv_path)
    
    #print(readWords("apple-twitter2.csv"))  #show list of words in rows
    print(readSentiments("apple-twitter2.csv")) #show sentiments from rows
    list2=[]
    list1 = readWords("apple-twitter2.csv")[0]  # take a list of words
    for i in (list1):                           # in all raws from file
        print (''.join(i).split())              # transform sentences into separate words
        list2 = ''.join(i).split()
    
    list3=list2[0]
    a=0
    for i in list3:
        a=a+ord(i)
       
    print(a)