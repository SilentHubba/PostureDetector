filename = "dataset.csv"

with open(filename, 'r') as f:
    first_line = f.readline()

with open(filename, 'w') as f:
    f.write(first_line)