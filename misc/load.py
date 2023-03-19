import os
import pickle

# Get the current working directory
cwd = os.getcwd()

# Specify the path to the pickle file
pickle_path = os.path.join(cwd, "data", "stock_data.pickle")

# Open the pickle file and load the data
with open(pickle_path, "rb") as f:
    stock_data = pickle.load(f)

