import pandas as pd

# Print the counts of the number of good and bad
# Posture data points to see how even it is for
# training
df = pd.read_csv("dataset.csv")
print(df['label'].value_counts())