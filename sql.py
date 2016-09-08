### SQL
#=====================================================================================================
from sqlalchemy import create_engine

# Creates an engine/connection to the SQL table
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/northwind')
# postgresql - SQL
# dsi_student - username
# gastudents - password
# @...amazonaws.com - server ip
# 5432 - port
# northwind - table/file name

# Saves SQL table into a pandas dataframe
pd.read_sql('SELECT * FROM table', con=engine)

sql = """
SELECT * FROM table
"""
pd.read_sql(sql)

## SQL commands
# Explanation
""""
CREATE TABLE - Create a new table
INSERT INTO - Insert a new row into a table
DELETE - Delete a row
UPDATE - Update the value in a row
SELECT - Select a row/or all(*)
DISTINCT - Return unique values only
FROM - Select a table
JOIN - Join another table (INNER JOIN, LEFT JOIN (default), RIGHT JOIN, FULL JOIN)
ON - Join another table on specific column
WHERE - Conditional format to extract specific criterions
LIKE - Conditional format to search for a specific pattern in a WHERE clause
GROUP BY - Group by a column of rows
HAVING - HAVING clause is used with GROUP BY, works like WHERE clause except for aggregate functions
ORDER BY - Order by a column of rows
LIMIT - Limit the number of entires
"""

# Examples
"""
CREATE TABLE table_name
(
column_name1 data_type(size),
column_name2 data_type(size),
column_name3 data_type(size),
....
);

INSERT INTO table_name (column1,column2,column3,...);

DELETE FROM table_name
WHERE some_column=some_value;

UPDATE table_name
SET column1=value1,column2=value2,...
WHERE some_column=some_value;

SELECT DISTINCT(table1.col1), table2.col1, table1.col2, SUM(table2.col2) FROM table1
JOIN table2
ON table1.col2=table2.col2
WHERE table2.col1 LIKE "s%"
GROUP BY table2.col2
HAVING table1.col1 > 10
ORDER BY table1.col1
LIMIT 5;
"""

# Cartesian Product
"""
SELECT col1, col2, col3, col4, col5, col6, col7, col8, col9 FROM table1, table2, table3
"""
#=====================================================================================================








