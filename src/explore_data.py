import pandas as pd

# Load the dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first 5 messages
print(data.head())

# Display the number of messages
print("\nTotal messages:")
print(len(data))

# Count spam and ham messages
print("\nMessage counts:")
print(data["label"].value_counts())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())