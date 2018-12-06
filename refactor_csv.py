import csv

# File path to the csv file.
csv_path = 'Apple-Twitter-Sentiment-DFE.csv'

# File path to the new csv file.
new_csv_file = 'apple-twitter.csv'

# Open the original csv file.
with open(csv_path, encoding="ISO-8859-1") as csv_file:
    # Create reader object used to iterate over every row in the file.
    csv_content = csv.reader(csv_file, delimiter=',')

    # Create/replace and write new csv file.
    with open(new_csv_file, mode='w') as new_csv_file:
        # Create writer object used to write rows in new csv file.
        csv_writer = csv.writer(new_csv_file, delimiter=',')

        for row in csv_content:
            # Write every 'sentiment' and 'text' to new row.
            csv_writer.writerow([row[5], row[11]])
