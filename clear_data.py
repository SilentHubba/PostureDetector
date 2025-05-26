# Remove all but the first line of the csv file.
# Do this to clear training data
filename = "dataset.csv"

with open(filename, 'r') as f:
    first_line = f.readline()

with open(filename, 'w') as f:
    f.write(first_line)