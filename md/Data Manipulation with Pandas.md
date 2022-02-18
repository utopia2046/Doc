# Data Manipulation with Pandas

## Panda Series

``` python
import pandas as pd

data = pd.Series([0.1, 0.2, 0.35, 0.5, 0.7, 1])
data.values
data.index

named_data = pd.Series([0.1, 0.2, 0.35, 0.5], index = ['a', 'b', 'c', 'd'])
named_data['b']
named_data['a':'c']
named_data[(named_data > 0.3) & (named_data < 0.4)]  # masking

log_data = pd.Series([0.01, 0.334, 0.16, -0.12], index = [1, 10, 100, 1000])
log_data[100]
log_data.loc[0:3]
log_data.iloc[0:3]

dictionary = pd.Series({'a': 1, 'b': 2, 'c':23})
```

## DataFrame

``` python
# build DataFrame from Series
states = pd.DataFrame({'state': names, 'population': population, 'area': area})
# build DataFrame from struct array
pd.DataFrame([{'a':1, 'b':2}, {'a':-1, 'c':0}])
# from 2-d NumPy array
pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
```

> Pandas also have pd.Penel for 3D, and pd.Panel4D for 4D data matrix. Index, loc, and iloc applies as well.

## Index

Pandas index are immutable array which supports unions, intersections, differences

``` python
indA = pd.Index[1, 2, 3, 5, 7, 9]
indB = pd.Index[2, 3, 6, 8, 10]
indA & indB   # intersection
indA | indB   # union
indA ^ indB   # symmetric difference
```

Pandas MultiIndex type allows us to query and aggregate data at different level

``` python
index = [('California', 2000), ('California', 2010),
    ('New York', 2000), ('New York', 2010)]
populations = [33871648, 37253956, 18976457, 19378102]

# Create multi-index from tuples
index = pd.MultiIndex.from_tuples(index)

# Create data Series
pop = pd.Series(populations, index=index)

pop
# California  2000    33871648
#             2010    37253956
# New York    2000    18976457
#             2010    19378102
# dtype: int64

# Select only 2010 data
pop[:, 2010]
# Select rows with population greater than 20000000
pop[pop > 20000000]

# Set names of multi-index levels
index = index.set_names(['region', 'year'])
pop = pop.reindex(index)
pop.sum(level='year')

# Append yet anther dimension to data
pop_df = pd.DataFrame({'total': pop, 'under18': [9267089, 9284094, 4687374, 4318033]})

# Calculate percentage of population under age 18
f_u18 = pop_df['under18'] / pop_df['total']

f_u18
# California  2000    0.273594
#             2010    0.249211
# New York    2000    0.247010
#             2010    0.222831
# dtype: float64
```

## Concatenation with joins

``` python
def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

# join with concat
df1 = make_df('ABC', [1, 2])
df2 = make_df('BCC', [3, 4])
df3 = pd.concat([df1, df2], sort=True)
df4 = pd.concat([df1, df2], sort=True, join='inner')

df1
#     A   B   C
# 1  A1  B1  C1
# 2  A2  B2  C2

df2
#     B   C
# 3  B3  C3
# 4  B4  C4

df3
#      A   B   C
# 1   A1  B1  C1
# 2   A2  B2  C2
# 3  NaN  B3  C3
# 4  NaN  B4  C4

df4
#     B   C
# 1  B1  C1
# 2  B2  C2
# 3  B3  C3
# 4  B4  C4
```

The pd.merge() function implements a number of types of joins: 1 to 1, 1 to many, and many to many.

``` python
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
    'supervisor': ['Carly', 'Guido', 'Steve']})

# merge function will automatically find columns with same name and use it as join key
# also you can specify the column name to join on
df5 = pd.merge(df1, df2)
df5 = pd.merge(df1, df2, on='employee') # same with previous line
df6 = pd.merge(df5, df3)
# also, there are 'left_on' and 'right_on' properties

# call set_index before merge
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)  # same with previous line

# merge also support left, right, inner, outer join
pd.merge(df6, df7, how='left')

# override column names on joining
pd.merge(df8, df9, on='name', suffixes=['_L', '_R'])

# select unique records
final.unique()

# use query function
final.query("year == 2010 & ages = 'total'")
```

## Aggregation and Grouping

``` python
import numpy as np
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': range(6),
    'data2': rng.randint(0, 10, 6)},
    columns=['key', 'data1', 'data2'])

# calculate sum of each key
df.groupby('key').sum()
# pass function names in aggregate()
df.groupby('key').aggregate(['min', np.median, 'max'])
# pass column names and function names in aggregate()
df.groupby('key').aggregate({'data1': 'min', 'data2': 'max'})
# define a filtering function
def filter_func(x):
    return x['data2'].std() > 4

# pass filter_func to filter out group A since its std is less than 4
df.groupby('key').std()
df.groupby('key').filter(filter_func)

# transform data using lambda function
df.groupby('key').transform(lambda x: x - x.mean())

# apply a function to the group result
def norm_by_data2(x):  # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

df.groupby('key').apply(norm_by_data2)
```

## Reading from CSV file

``` python
# read data from a csv file
data = pd.read_csv("titanic.csv")
data.head()
# drop the name column since it doesn't seems relevant
data.drop(data.columns[[3]], axis=1, inplace=True)

# Series also has a from_csv method
Series.from_csv('example.csv', parse_dates=True)

# And there is a Python native csv module
import csv
f = open('example.csv')
class my_dialect(csv.Dialect):
    lineterminator = "\n"
    delimiter = ";"
    quotechar = '"'
    doublequote = True
csv.reader(f, dialect=my_dialect)
```

## Reading from JSON file

``` python
f = open("example.json")
s = f.read()
import json
# json string to object
obj = json.loads(s)
# object to json string
json.dumps(obj)
```

## Parse HTML table data from Web

Reference:
<https://docs.python.org/3/howto/urllib2.html>

``` python
from lxml.html import parse
from urllib import request
parsed = parse(request.urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
links = doc.findall('.//a') # select nodes using XPath selector
links2 = doc.cssselect('a') # select nodes using css selector

# get all links targets
urls = [link.get('href') for link in doc.findall('.//a')]

# get call table data
tables = doc.cssselect('table.list-options')
call_table = tables[0]
headers = [th.text_content() for th in call_table.cssselect('thead th')]
rows = call_table.cssselect('tbody tr')
# read a row
def unpack(row):
    return [td.text_content() for td in row.cssselect('td')]
# read all rows and create a DataFrame
cells = [unpack(row) for row in rows]
data = pd.DataFrame(cells, columns=headers)
data.head()
```

## Get data from ODATA service

``` python
import requests
url = 'http://search.twitter.com/search.json?q=python%20pandas'
resp = requests.get(url)

import json
data = json.loads(resp.text)
```

## Parse XML data

``` python
from lxml import objectify
parsed = objectify.parse(open('UserConnections.xml'))
root = parsed.getroot()

data = [{'Name': server.Name,
    'Details': server.Details,
    'ConnectionString': server.ConnectionString} for server in root.getchildren()]
```

## Parse Excel data

``` python
!pip install xlrd --upgrade
!pip install openpyxl --upgrade

xls = pd.ExcelFile('Segment_Summary_20180302.xls')
table = xls.parse('Sheet1')
```

## Get Data from SQL

``` python
import pyodbc

sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER=cashew;DATABASE=Vowper;Trusted_Connection=yes;')

# read
query = """SELECT TOP 100 word
    FROM [Vowper].[dbo].[WordFrequencyIndex]
    WHERE frequency_index > 3056
    ORDER BY [frequency_index] ASC"""

df = pd.read_sql(query, sql_conn)
df.head()

# insert
df = pd.DataFrame([[233, 'AAA'], [666, 'BBB']], columns=['ID', 'Name'])
cursor = sql_conn.cursor()
for index,row in df.iterrows():
    cursor.execute("INSERT INTO dbo.Test([ID],[Name]) VALUES (?,?)",
        row['ID'], row['Name'])
    sql_conn.commit()
cursor.close()
sql_conn.close()
```

## Get Data from MongoDB

Reference:
<https://www.w3schools.com/python/python_mongodb_getstarted.asp>

``` python
import pymongo

# Create a database called "mydatabase"
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]

# Check if "mydatabase" exists
dblist = myclient.list_database_names()
if "mydatabase" in dblist:
  print("The database exists.")

# Create collection
mycol = mydb["customers"]

# Insert
mydict = {"name":"John", "address":"Highway 37"}
x = mycol.insert_one(mydict)
print(x.inserted_id)
mylist = [
  { "name": "Amy", "address": "Apple st 652"},
  { "name": "Hannah", "address": "Mountain 21"},
  { "name": "Viola", "address": "Sideway 1633"}
]
x = mycol.insert_many(mylist)

# Find & update
x = mycol.find_one()
for x in mycol.find():
  print(x)
for x in mycol.find({},{ "_id": 0, "name": 1, "address": 1 }):
  print(x)
mydoc = mycol.find().sort("name", -1)

myquery = { "address": "Park Lane 38" }
mycol.delete_one(myquery)
mydoc = mycol.find(myquery)

myquery = { "address": { "$gt": "S" } }
mydoc = mycol.find(myquery)
x = mycol.delete_many(myquery)

myquery = { "address": { "$regex": "^S" } }
newvalues = { "$set": { "name": "Minnie" } }
x = mycol.update_many(myquery, newvalues)
print(x.modified_count, "documents updated.")
```

### Useful parameters of read_csv function

Parameter | Meaning
----------|-----------------------------------------------------------------------------
header    | line number as column name, default=0; pass header=None if no header in file
index_col | index column number
names     | assign string array as column names
skiprows  | skip top n rows
na_values | replace NA value
comment   | comment characters
nrows     | read first n rows
encoding  | for example 'utf-8'

### Write to a csv file

``` python
data.to_csv('output.csv', index=False)
```

## Grouping

Aggregating similar like GROUP BY but more powerful and fast.

``` python
# select only Name and Age column
data.loc[:, ['Name', 'Age']]
# select certain row with condition
data.loc[data['Name'] == 'Braund, Mr. Owen Harris']
data.groupby(['Sex', 'Survived'])['PassengerId'].count()
data.groupby('Pclass')['Age'].mean()
meanAge = data.groupby('Pclass')['Age'].mean().reset_index()
```

## Data cleansing

1. Remove unnecessary columns;
2. Remove duplicate or empty data;
3. Fill missing data, update incorrect data and incorrect data format;

Pandas data cleansing functions

* isnull()
* dropna()
* fillna()
* duplicated()
* drop_duplicates()
* set_index()

``` python
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data.dropna()
data.to_csv('cleansed.csv', sep=',')
```

## Build a model using scikit models

Building models using Support Vector Classifier (SVC) & Decision Tree Classifier

``` python
X = data.drop('Survived', axis=1)
y = data['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Train a SVC model
from sklearn.svm import SVC
model_svc = SVC(kernel='linear', probability=True, random_state=0)
model_svc.fit(X_train, y_train)

# Evaluate SVC model
y_predict = model_svc.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)

# Train a tree model
from sklearn import tree
model_tree = tree.DecisionTreeClassifier(min_samples_split=15)
model_tree.fit(X_train, y_train)

y_predict = model_tree.predict(X_test)
accuracy_score(y_test, y_predict)
```

## Pivot Tables

``` python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic.head()

# survive rate by sex and class using groupby and aggregate function
titanic.groupby('sex')[['survived']].mean()
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

# same aggregation using pivot table
titanic.pivot_table('survived', index='sex', columns='class')

# add age as 3rd dimension
age = pd.cut(titanic['age'], [0, 18, 60, 80])
titanic.pivot_table('survived', ['sex', age], 'class')

# add fare as 4th dimension
fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

# additional pivot_table options
# full signature of pivot_table function is as below
# DataFrame.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All')
titanic.pivot_table(index='sex', columns='class', aggfunc={'survived':sum, 'fare':'mean'})
# show totals
titanic.pivot_table('survived', index='sex', columns='class', margins=True)
```

## String Operations

### str Object Built-in functions

* len()
* lower()
* upper()
* islower()
* isupper()
* startswith()
* endswith()
* isnumeric()
* isdecimal()
* isalpha()
* isdigit()
* index()
* find()
* rfind()
* partition()

### Pandas string methods

* get()
* slice()
* slice_replace()
* cat()             # concatenation
* repeat()
* normalize()       # return Unicode form
* pad()
* wrap()
* join()

### Regular expressions

* match()
* extract()
* findall()
* replace()
* contains()
* count()
* split()
* rsplit()

``` python
# extract alphabetic characters
monte.str,extract('([A-Za-z]+)')
# find all word start and end with a consonant
monte.str.findall(r'^[^AEIOU].*[^]aeiou'$)  # ^ for start of string, $ for end
```

``` python
# example: Recipe Database
!curl -O https://s3.amazonaws.com/openrecipes/20170107-061401-recipeitems.json.gz

import json
import pandas as pd
import numpy as np

with open('20170107-061401-recipeitems.json', 'r', encoding='utf-8') as f:
    head = [next(f) for x in range(2000)]
    array = []
    for d in head:
        j = json.loads(d)
        array.append(j)
recipes = pd.DataFrame(array)

recipes.ingredients.str.len().describe()
recipes.name[np.argmax(recipes.ingredients.str.len())]
recipes.ingredients.str.contains('[Cc]innamon').sum()
```

To erase a variable from iPython session, use the magic command:
`%reset_selective <regular_expression>`

### Demo: Reading JSON files from subfolders into Pandas DataFrame

``` python
""" JSON Demo """
import pandas as pd
import os
import json

# Example usage of from_records method
records = [("Espresso", "5$"),
           ("Flat White", "10$")]

pd.DataFrame.from_records(records)

pd.DataFrame.from_records(records,
                          columns=["Coffee", "Price"])

#####
KEYS_TO_USE = ['id', 'all_artists', 'title', 'medium', 'dateText',
               'acquisitionYear', 'height', 'width', 'units']

def get_record_from_file(file_path, keys_to_use):
    """ Process single json file and return a tuple
    containing specific fields."""

    with open(file_path) as artwork_file:
        content = json.load(artwork_file)

    record = []
    for field in keys_to_use:
        record.append(content[field])

    return tuple(record)

# Single file processing function demo
SAMPLE_JSON = os.path.join('..', 'collection-master',
                           'artworks', 'a', '000',
                           'a00001-1035.json')

sample_record = get_record_from_file(SAMPLE_JSON,
                                     KEYS_TO_USE)

def read_artworks_from_json(keys_to_use):
    """ Traverse the directories with JSON files.
    For first file in each directory call function
    for processing single file and go to the next
    directory.
    """
    JSON_ROOT = os.path.join('..', 'collection-master',
                             'artworks')
    artworks = []
    for root, _, files in os.walk(JSON_ROOT):
        for f in files:
            if f.endswith('json'):
                record = get_record_from_file(
                            os.path.join(root, f),
                            keys_to_use)
                artworks.append(record)
            break

    df = pd.DataFrame.from_records(artworks,
                                   columns=keys_to_use,
                                   index="id")
    return df

df = read_artworks_from_json(KEYS_TO_USE)
```

## DateTime

``` python
# create a datetime object
from datetime import datetime
datetime(year=2015, month=7, day=4)
datetime.datetime(2015, 7, 4, 0, 0)

# parse datetime from string
from dateutil import parser
date = parser.parse("4th of July, 2015")

# date to string
date.strftime('%A')

# using NumPy's datetime64
import numpy as np
date = np.array('2015-07-04', dtyle=np.datetime64)

# date and time in Pandas
import pandas as pd
date = pd.to_datetime("4th of July, 2015")
Timestamp('2015-07-04 00:00:00')

date + pd.to_timedelta(np.arange(12), 'D')
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8, freq='H')
pd.timedelta_range(0, periods=10, freq='H')
```

### Pandas frequency codes

| Code | Description   |
| ---- | ------------- |
| D    | Calendar day  |
| B    | Bussiness day |
| W    | Weekly        |
| M    | Month end     |
| Q    | Quarter end   |
| A    | Year end      |
| H    | Hour          |
| T    | Minutes       |
| S    | Seconds       |
| L    | Milliseconds  |
| U    | Mocroseconds  |
| N    | Nanoseconds   |
