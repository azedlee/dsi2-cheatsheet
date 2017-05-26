### Learn about the data
#=====================================================================================================
df.info() # Tells you the shape and the types
df.shape # Tells you the shape of the dataframe
df.dtypes # Tells you the data type for each column
df.head() # Prints out first 5 rows
df.tail() # Prints out last 5 rows
df.columns # Prints out all column names
df['col1'].unique() # Prints out all unique values in col1
df['col1'].value_counts() # Prints out each unique value and # of instances
#=====================================================================================================










### Clean the data
#=====================================================================================================
## Change Yes/No, True/False into 0 and 1
df['col1'].map(lambda x: 1 if x == 'Yes' else 0)

## Remove all $ signs, or any strings, in dataframes and change to a float, if needed, with .apply
def remove_dollar_sign(x):
	if '$' in x:
		return float(x.replace('$', ''))
	else:
		return x

df['col'].apply(remove_dollar_sign)

# Simple way to change all values in column to different type with .apply
df['col'].apply(float)

## Rename columns
df.rename(columns={'col1':'new_col1', 'col2':'new_col2'}, inplace=True)

# Another way
update_columns = ['new_col1', 'new_col2']
df.columns = update_columns
# or
new_df = pd.DataFrame(data=df, columns=update_columns)

## Drop/Edit Nan Values
# Drop rows with NaN values
df.dropna()
df['col1'].dropna()
# Drops NaN from ONLY col1
df = df.dropna(subset=['col1'])

# Edit NaN Values
df.fillna('new_value')
df['col1'].fillna('new_value')

## Create new column
df['new_col'] = np.mean(df['col1']) / np.mean(df['col2'])

## Masking
# new_col has to be True AND col1 has to be 0 OR col2 does not equal to 'you'
new_mask = (df['new_col'] == True) & (df['col1'] == 0) | (df['col2'] != 'you')
df[new_mask]

## Quick way to create new dataframe with select columns from another dataset
new_df = df[['col1', 'col2', 'col3']]

## Dealing with outliers
# Removing
def reject_outliers(data, m=1.5):
    data[abs(data - np.mean(data)) > m * np.std(data)]

# Dropping columns that start with 'objective'
df.drop([c for c in df.columns if c.startswith('objective')], axis=1, inplace=True)

## Indexing
"""
.loc - indexes with the labels for rows and columns
.iloc - indexes with the integer positions for rows and columns
.ix - indexes with both lebals and integer positions
"""

# Removing HTML tags
import re

def remove_html(x):
    clean_html = re.compile('<.*?>')
    try:
        cleaned = re.sub(clean_html, '', x)
        return cleaned
    except:
        return x


## Pivot Tables - Long to Wide
df_wide = pd.pivot_table(df_long, # The Data frame you want to convert
                        columns=['col'], # The values in the long df you want to assign for the wide dataframe
                        values='value', # The values in the long df you want to pivot to the wide dataframe
                        index=['subject_id'], # The columns in the long df you want to become the index for the wide dataframe
                        aggfunc=np.mean, # Aggregate function that defaults to the mean, can put own function in, works like .apply 
                        fill_value=np.nan) # Fills in all empty values as assigned value

# Pivot Table example
   A   B   C      D
0  foo one small  1
1  foo one large  2
2  foo one large  2
3  foo two small  3
4  foo two small  3
5  bar one large  4
6  bar one small  5
7  bar two small  6
8  bar two large  7

table = pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value='hi')

# Output:
          small  large
foo  one  1      4
     two  6      'hi'
bar  one  5      4
     two  6      7

## Melt() - Wide to Long
df_long = pd.melt(df_wide, # The Data frame you want to convert
                  id_vars=['col1','col2'], # The identifiers for the other columns
                  value_vars=, # The value that identifies to each id_vars
                  var_name=, # The column name for value_vars
                  value_name=) # The column name for the values for each value_vars

# Melt example
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})

   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6

pd.melt(df, id_vars=['A'], value_vars=['B'], var_name='myVarname', value_name='myValname')

# Output:
   A myVarname  myValname
0  a         B          1
1  b         B          3
2  c         B          5


## Merging
new_df = df1.merge(df2, on='Id', how='left')
new_df = pd.merge(df1, df2, on='Id', how='right')
# Multiple merges in 1 line
new_df = df1.merge(df2, on='Id', how='left').merge(df3, on='Name', how='inner').merge(df4, on='Password', how='outer')
#=====================================================================================================









### Time Series
#=====================================================================================================

# Convert to datetime
pd.to_datetime(Feature)

# Make index the date
df.index = df['date']

from datetime import datetime

# Time this lesson plan was written
# Year, Month, Date, Hour, Minute, Second, Micro-Second
lesson_date = datetime(2016, 3, 5, 23, 31, 1, 844089)

# Output date time variables
lesson_date.day, lesson_date.month, lesson_date.year, lesson_date.hour

# Prints the datetime of what is happening
datetime.now()


#=====================================================================================================
