## function pulled from the neo package. Reads a csv file as a dict.
import csv

def read_csv_as_dict(filename, delimiter=None):
    # csv_data = open(filename, 'r')
    if delimiter is None:
        if filename.suffix == '.csv':
            delimiter = ','
        elif filename.suffix == '.tsv':
            delimiter = '\t'
        else:
            raise ValueError("Function parses only .csv or .tsv files")

    with open(filename, 'r') as csv_data:
        data = list(csv.reader(csv_data, delimiter=delimiter))
        # get list of dict with keys as first row of csv
        keys = data[0]
        data = data[1:]
        list_of_dict = [dict(zip(keys, row)) for row in data]

        # ind_dict = dict(zip(data[0],get_mulList(*data[1:])))
    return list_of_dict





