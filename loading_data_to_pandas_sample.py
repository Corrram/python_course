import sqlite3

import pandas as pd

# loading sql table
engine = sqlite3.connect("music.db")
df = pd.read_sql_query("SELECT * FROM artists", engine)
folder_path = r"C:\Users\corra\Documents\repositories"
df.to_excel(folder_path + "/test.xlsx", index=False)

# loading csv file from internet
url_string = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url_string)
df.groupby("sex")["survived"].mean()
print("Done!")
