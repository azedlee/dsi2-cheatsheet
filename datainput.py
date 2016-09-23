### Uploading the CSV file
#=====================================================================================================
import pandas as pd

df = pd.read_csv('../../../example.csv')
df = pd.read_json('http://api.github.com')
df = pd.read_sql('SELECT * FROM table', con=engine)
#=====================================================================================================