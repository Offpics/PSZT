import csv

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

if __name__ == "__main__":
    # File path to the csv file.
    csv_path = 'Apple-Twitter-Sentiment-DFE.csv'

    # File path to the new csv file.
    new_csv_path = 'apple-twitter.csv'

    # Refactor dataset.
    refactor(csv_path, new_csv_path);