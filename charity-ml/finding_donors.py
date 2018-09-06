# Importing libraries necessary for this project
import pandas as pd
from IPython.display import display


if __name__ == '__main__':
    # Loading dataset
    in_file = 'census.csv'
    full_data = pd.read_csv(in_file)

    display(full_data.head())

    # Making some data exploration
    n_records = len(full_data)
    n_greater_50k = len(full_data[full_data.income == '>50K'])
    n_at_most_50k = len(full_data[full_data.income == '<=50K'])
    greater_percent = n_greater_50k / n_records * 100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {0:.2f}%".format(greater_percent))
