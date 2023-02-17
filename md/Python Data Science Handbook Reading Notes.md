# Python Data Science Handbook Reading Notes

- [Python Data Science Handbook Reading Notes](#python-data-science-handbook-reading-notes)
  - [IPython](#ipython)
    - [Start Anaconda prompt](#start-anaconda-prompt)
    - [Start ipython](#start-ipython)
    - [Built-in help](#built-in-help)
    - [Magic commands](#magic-commands)
    - [In \& Out objects array](#in--out-objects-array)
    - [Run a shell command](#run-a-shell-command)
    - [Errors and debugging](#errors-and-debugging)
  - [NumPy library](#numpy-library)
    - [UFuncs](#ufuncs)
    - [Common Statistic functions](#common-statistic-functions)
    - [Broadcasting](#broadcasting)
    - [Use boolean arrays as mask](#use-boolean-arrays-as-mask)
    - [Bitwise Operators](#bitwise-operators)
    - [Fancy Indexing](#fancy-indexing)
    - [Sorting](#sorting)
  - [Pandas](#pandas)
    - [Pandas Series, DataFrame \& Index](#pandas-series-dataframe--index)
    - [Multi-level Indexing](#multi-level-indexing)
    - [Merging](#merging)
    - [Aggregation](#aggregation)
    - [Pivot table](#pivot-table)
    - [Pandas String Operations](#pandas-string-operations)
    - [Time Series](#time-series)
    - [Eval](#eval)
  - [Visualization](#visualization)
    - [Matplotlib](#matplotlib)
      - [Line Styles \& Markers](#line-styles--markers)
      - [Colors](#colors)
      - [Plot properties](#plot-properties)
      - [Common pyplot functions](#common-pyplot-functions)
      - [Common graphics types](#common-graphics-types)
      - [Set style](#set-style)
      - [3D plot](#3d-plot)
    - [Pandas plot](#pandas-plot)
    - [Bokeh](#bokeh)
    - [Plotly](#plotly)
  - [Machine Learning](#machine-learning)
    - [Scikit-Learn](#scikit-learn)

## IPython

### Start Anaconda prompt

``` shell
%windir%\System32\cmd.exe "/K" C:\Users\jiew\Anaconda3\Scripts\activate.bat C:\Users\jiew\Anaconda3
```

### Start ipython

``` shell
ipython.exe
```

### Built-in help

``` python
help(len)
len?        # get function inline docString
len??       # get function source code
str.*find*? # find all functions with 'find' string
# https://docs.python.org/3/library/inspect.html
import inspect
inspect.getdoc
inspect.getfile
inspect.getsource
```

> press TAB for auto-completion

### Magic commands

``` python
%paste  # paste multiple line code with indent
%run    # run a .py file
%magic  # show all magic command

%history -n 1-5 # show latest 5 commands

%time   # time the execution of a single statement
%timeit # time repeated execution of a single statement for more accuracy
%prun   # run code with the profiler
%lprun  # run code with line-by-line profiler
%memit  # measure the memory usage of a single statement
% mprun # run code with line-by-line memory profiler
```

### In & Out objects array

- `In` and `Out` are built-in object arrays for history
- Previously output could be accessed using `_`, `__`, `___`

### Run a shell command

``` python
!dir
!mkdir myproject
```

### Errors and debugging

``` python
%xmode Plain
%xmode Verbose
%debug
?
```

## NumPy library

<https://numpy.org/>

```python
import numpy as np
# generate matrix
np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((2, 3), 3.14)
np.eye(3)
np.arange(0, 10, 2)   # array([0, 2, 4, 6, 8]), _[-1] = 8
np.linspace(0, 1, 5)  # array([0., 0.25, 0.5, 0.75, 1.])
np.random.random((3, 4))         # uniformly distributed
np.random.normal(0, 1, (3, 4))   # normally distributed
np.random.randint(0, 10, (3, 4)) # randint(low, high=None, size=None, dtype=int)
np.empty(3)           # create an uninitialized array
# slice sub array
x2 = x[:, ::2]        # all rows, every other column
x3 = x[::-1, ::-1]    # start:end:step, upside-down and reversed in every row
# reshape
grid = np.arange(1, 10).reshape((3, 3))
```

> Notice that NumPy array slice DONT copy the array, it is just a sub-view. If you modify an element value, the value in slice will also change. If you want to get a copy of the data slice, you can use `copy()` method. Modifing the copied data won't change original data.

``` python
# concatenation functions
np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
np.vstack(tup)
np.hstack(tup)
# split functions
np.split(ary, indices_or_sections, axis=0)
upper, lower = np.vsplit(ary, indices_or_sections)
left, right = np.hsplit(ary, indices_or_sections)
```

### UFuncs

Since Python loops could be very slow due to dynamic type check and dispatching, it is recommended to **use UFuncs as much as possible**.

UFuncs:

- Array arithmetic: `+(add), -(subtract), -(negative), *(multiply), /(divide), **(power), //(floor_divide), % (mod), abs`
- Trigonometric functions: `sin, cos, tan, arcsin, arccos, arctan`
- Exponents and logarithms: `exp(e^x), exp2(2^x), power, log = ln(x), log2, log10, expm1 = exp(x)-1, log1p = log(1+x)`
- Hyperbolic trig functions
- Bitwise arithmetic: `logical_and, logical_or, logical_xor, &, |, ^`
- Comparison operators: `greater, greater_equal, less, less_equal, equal, not_equal, >, >=, <, <=, ==, !=`
- Conversions from radians to degrees
- Rounding and remainders
- Other functions defined in scipy.special, ref: <https://docs.scipy.org>
- Aggregates: `reduce, accumulate, sum, min, max, mean, prod, std, var, any, all`

``` python
x = np.arange(1, 6)
np.add.reduce(x)        # sum of each element, same as np.sum(x)
np.multiply.reduce(x)   # production of each element
np.multiply.outer(x, x) # a 5x5 matrix of pair-wise production

x = np.random.randn(10)
y = np.random.randn(10)
np.where(x > y, x, y)   # choose each element the bigger one between x and y
```

``` python
import pandas as pd
data = pd.read_csv('data/president_heights.csv')
data.head(5)
data.describe()
heights = np.array(data['height(cm)']) # get data height column as np array
heights.max()
heights.min()
heights.mean()
# simple line chart with matplot
%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlable('height (xm)')
plt.ylable('number')
plt.show()
```

<https://matplotlib.org/stable/tutorials/introductory/pyplot.html>

### Common Statistic functions

- sum, prod
- min, max: optional parameter to specify which dimension, e.g. x.min(axis=0)
- argmin, argmax: index of min/max
- mean, std, var, median, percentile
- any, all: evaluate whether any/all elements are true
- unique(x), intersect1d(x, y), union1d(x, y), in1d(x, y), setdiff1d(x, y), setxor1d(x, y): basic set operations

### Broadcasting

When 2 array of different dimesion are in same arithmetic, NumPy is trying to extend the array size so that they could match the same bigger size. Rules are:

1. If 2 arrays differ in their number of dimensions, the shape of the one with fewer dimensions is **padded** with ones on its leading (left) side;
2. If the shape of 2 arrays do not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape;
3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

``` python
# example: use broadcasting to compute the z = f(x, y) across the grid
x = np.linspace(0, 5, 50)                 # x.shape is (50,)
y = np.linspace(0, 5, 50)[:, np.newaxis]  # y.shape is (50, 1)
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0,5,0,5])
plt.show()

# example: use broadcasting to normalize data
x = np.random.random((10,3))
x_mean = x.mean(0)   # calculate mean by axis 0
x_centered = x - x_mean
x_centered.mean(0)
```

### Use boolean arrays as mask

``` python
# example: inches is an array of each day's rain amount in Seatle, 2014
rainy = (inches > 0)    # rainy is a boolean array indicating if a day is a rainy day
summery = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0) # 90 days after Jun 21st
print("median precip on rainy days", np.median(inches[rainy]))
print("maximum precip on summer rainy days", np.max(inches[rainy & summer]))

# count with query (boolean array, Comparison Operators as ufuncs)
np.count_nonzero(x < 6)
np.sum((x > 5) & (x < 10))
np.any(x > 10)
np.all(x > 0, axis=1)
np.sum((inches > 0.5) & (inches < 1))
np.sum(~( (inches <= 0.5) | (inches >= 1) )) # 0.5 < inches < 1
```

### Bitwise Operators

Operator | Equivalent ufunc
---------|-----------------
`&`      | np.bitwise_and
`|`      | np.bitwise_or
`^`      | np.bitwise_xor
`~`      | np.bitwise_not

### Fancy Indexing

Pass a single list or array of indices to obtain certain elements

``` python
ind = [3, 7, 4]
x[ind]        # same as [x[3], x[7], x[2]]

X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]   # same as [X[0, 2], X[1, 1], X[2, 3]]

# example: choose random subset
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100) # X.shape is (100, 2)
indices = np.random.choice(X.shape[0], 20, replace=False) # 20 random indeces with no repeat
selection = X[indices]  # select 20 random points
```

### Sorting

``` python
arr = np.array(range(0,100))  # [0, 99]
np.random.shuffle(arr)        # shuffle arr
arr.sort()                    # re-order arr
```

## Pandas

### Pandas Series, DataFrame & Index

Pandas Series has an explicitly defined index associated with the values, you can think of a Pandas Series a bit like a specialization of a Python dictionary.

DataFrame is an analog of a two-dimensional array with both flexible row indices and flexible column names.

Index object can be thought of either as an **immutable** array or as an ordered set。

``` python
# create Pandas Series from Python dictionary
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)

area_dict = {'California': 423967,
             'Texas': 695662,
             'New York': 141297,
             'Florida': 170312,
             'Illinois': 149995}
area = pd.Series(area_dict)

# create DataFrame from 2 Series
states = pd.DataFrame({'population': population,
                       'area': area})
states['density'] = states['population'] / states['area']
states.density         # get one column
states.loc['Texas']    # get one row
data.loc[data.density > 100, ['pop', 'density']]  # get subset using fancy index and bitmask

# DataFrame constructor signature
pd.DataFrame(
    data=None,
    index: 'Axes | None' = None,
    columns: 'Axes | None' = None,
    dtype: 'Dtype | None' = None,
    copy: 'bool | None' = None,
)

# Indeces support union, intersect, difference
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB  # intersection
indA | indB  # union
indA ^ indB  # symmetric difference

# fill NaN value when index is missing
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B                   # missing indeces will be filled as NaN
A.add(B, fill_value=0)  # fill missing indeces with 0
```

Python operators and their equivalent Pandas object methods:

Python Operator | Pandas Method(s)
----------------|---------------------------
`+`             | add()
`-`             | sub(), subtract()
`*`             | mul(), multiply()
`/`             | truediv(), div(), divide()
`//`            | floordiv()
`%`             | mod()
`**`            | pow()

NaN is specifically a floating-point value; there is no equivalent NaN value for integers, strings, or other types

``` python
# get max/min value of integer and float type
# https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html
# https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
np.iinfo(np.int32).min
np.iinfo(np.int32).max
np.iinfo(np.int64).max    # 9223372036854775807
np.finfo(np.float64).max  # 1.7976931348623157e+308
np.finfo(np.double).max   # 1.7976931348623157e+308
```

Useful methods to handle missing values

- `isnull()`: Generate a boolean mask indicating missing values
- `notnull()`: Opposite of isnull()
- `dropna()`: Return a filtered version of the data
- `fillna()`: Return a copy of the data with missing values filled or imputed

### Multi-level Indexing

Just as we were able to use multi-indexing to represent two-dimensional data within a one-dimensional Series, we can also use it to represent data of three or more dimensions in a Series or DataFrame. Each extra level in a multi-index represents an extra dimension of data.

``` python
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
index = pd.MultiIndex.from_tuples(index)
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
# pop = pop.reindex(index)
pop[:, 2010]
# unstack multi-level series to DataFrame
pop_df = pop.unstack()
# stack DataFrame to multi-level series
pop_df.stack()

# example: health data
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# create the DataFrame and calculate average value by level
health_data = pd.DataFrame(data, index=index, columns=columns)
data_mean = health_data.mean(level='year')
data_mean.mean(axis=1, level='type')
```

Reference: <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#swapping-levels-with-swaplevel>

### Merging

<https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>

<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html#pandas.DataFrame.join>

Common functions in Pandas to combine DataFrames are:

- concat()
- append()
- merge()
- join()

``` python
# Pandas functions signature
pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)[source]

pandas.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)

DataFrame.append(other, ignore_index=False, verify_integrity=False, sort=False)

DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None)

DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)

# Examples: join 2 DataFrame on specified key
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue', 'Aaa'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR', '']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})

# pandas is smart enough to find the mutual key 'employee' and join on it
pd.merge(df1, df2)
# specify join key explicitly
pd.merge(df1, df2, on='employee', how='inner')
pd.merge(df1, df3, left_on='employee', right_on='name', how='right').drop('name', axis=1)

# check diff of 2 lists
listA = ['a', 'b', 'c']
listB = ['a', 'c', 'd']
dfa = pd.DataFrame({'id': listA})
dfb = pd.DataFrame({'id': listB})
# find in merged list where _merge is not both
pd.merge(dfa, dfb, indicator=True, how='outer').loc[lambda x : x['_merge']!='both']
# equals to
pd_merge = pd.merge(dfa, dfb, indicator=True, how='outer')
pd_merge[pd_merge['_merge']!='both']
# or you can use concat and drop_duplicates but it won't save which record is from where
diff = pd.concat([dfa,dfb]).drop_duplicates(keep=False)

# example: US states data
pop = pd.read_csv('data/state-population.csv')
areas = pd.read_csv('data/state-areas.csv')
abbrevs = pd.read_csv('data/state-abbrevs.csv')

# pop.head()
  state/region     ages  year  population
0           AL  under18  2012   1117489.0
1           AL    total  2012   4817528.0
2           AL  under18  2010   1130966.0
3           AL    total  2010   4785570.0
4           AL  under18  2011   1125763.0

# areas.head()
        state  area (sq. mi)
0     Alabama          52423
1      Alaska         656425
2     Arizona         114006
3    Arkansas          53182
4  California         163707

# abbrevs.head()
        state abbreviation
0     Alabama           AL
1      Alaska           AK
2     Arizona           AZ
3    Arkansas           AR
4  California           CA

# merge population and abbrevs
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1) # drop duplicate info
merged.head()
# check and fill missing values
merged.isnull().any()
merged[merged.population.isnull()]
merged.loc[merged['state'].isnull(), 'state/region'].unique()
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
# merge areas and drop total
final = pd.merge(merged, areas, on='state', how='left')
final.head()
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace=True)
final.head()
# query data
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
density.head()
```

### Aggregation

<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html>

Split-Apply-Combine
aggregate, filter, transform, apply

Common built-in Pandas aggregations for Series and Data Frame:

Aggregation      | Description
-----------------|--------------------------------
count()          | Total number of items
first(), last()  | First and last item
mean(), median() | Mean and median
min(), max()     | Minimum and maximum
std(), var()     | Standard deviation and variance
mad()            | Mean absolute deviation
prod()           | Product of all items
sum()            | Sum of all items

``` python
# example: seaborn planets dataset
planets = pd.read_csv('data/planets.csv')
planets.groupby('method')['orbital_period'].describe()
# iterate through each groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))
# more aggregate options via aggregate function
planets.groupby('year').aggregate(['mean', np.median, max])
planets[planets['year'] > 2000].groupby('year').count()
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
# more filter, transform, apply examples
df.groupby("B").filter(lambda x: len(x) > 2, dropna=False)
df.groupby('key').transform(lambda x: x - x.mean())
planets.groupby('year').apply(lambda x: x.describe())
```

### Pivot table

<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html>

``` python
# The definition of pandas.pivot_table function is:
pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)
```

Parameters

- `data` : DataFrame
- `values` : column to aggregate, optional
- `index` : column, Grouper, array, or list of the previous
- `columns` : column, Grouper, array, or list of the previous
- `aggfunc` : function, list of functions, dict, default numpy.mean
- `fill_value` : scalar, default None, Value to replace missing values with
- `margins` : boolean, default False, Add all row / columns (e.g. for subtotal / grand totals)
- `dropna` : boolean, default True, Do not include columns whose entries are all NaN
- `margins_name` : string, default 'All', Name of the row / column that will contain the totals when margins is True.

``` python
titanic = pd.read_csv('data/titanic.csv')
# survice rate by sex and class
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
# cut age in phrases
age = pd.cut(titanic['age'], [0, 18, 60, 80])
fare = pd.qcut(titanic['fare'], 3)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
titanic.pivot_table('survived', ['sex', age], 'class')
```

Example:
<https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/>

``` python
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.parser import parse

data = pd.read_csv('./dataset/phone_data.csv')
# Convert date from string to date times
data['date'] = data['date'].apply(parse, dayfirst=True)
data.head(3)
#    index                date  duration  item    month   network network_type
# 0      0 2014-10-15 06:58:00    34.429  data  2014-11      data         data
# 1      1 2014-10-15 06:58:00    13.000  call  2014-11  Vodafone       mobile
# 2      2 2014-10-15 14:46:00    23.000  call  2014-11    Meteor       mobile

# How many rows the dataset
data['item'].count()

# What was the longest phone call / data entry?
data['duration'].max()

# How many seconds of phone calls are recorded in total?
# SUM(duration) WHERE item == 'call'
data['duration'][data['item'] == 'call'].sum()

# How many seconds of each type?
# SUM(duration) GROUP BY item
data.groupby('item').sum()
#        index  duration
# item
# call  159160  92321.00
# data   69774   5164.35
# sms   115101    292.00

# How many entries are there for each month?
# COUNT(month)
data['month'].value_counts()
# 2014-11    230
# 2015-01    205
# 2014-12    157
# 2015-02    137
# 2015-03    101
data.groupby(['month']).groups.keys()
# dict_keys(['2014-11', '2014-12', '2015-01', '2015-02', '2015-03'])

# Number of non-null unique month entries
data['month'].nunique()

# How many entries has month '2014-11'
# COUNT(index) WHERE month == '2014-11'
len(data.groupby(['month']).groups['2014-11'])
len(data[data['month']=='2014-11'])

# List first entry of each month group
data.groupby('month').first()

# Get the sum of the durations per month
data.groupby('month')['duration'].sum()

# Get the number of dates / entries in each month
data.groupby('month')['date'].count()

# What is the sum of durations, for calls only, to each network
# SUM duration GROUP BY network WHERE item == 'call'
data[data['item'] == 'call'].groupby('network')['duration'].sum()

# How many calls, sms, and data entries are in each month?
# COUNT(date) GROUP BY month, item
data.groupby(['month', 'item'])['date'].count()

# Result type
result = data.groupby('month')['duration'].sum() # produces Pandas Series
type(result)
result = data.groupby('month')[['duration']].sum() # Produces Pandas DataFrame
type(result)

# Tell pandas don't set index on the group key
data.groupby('month', as_index=False).agg({"duration": "sum"})

# Specify aggregation methods using agg() function
aggregation = {
    'duration': { # aggregate value column
        'total_duration': 'sum', # specify aggregation method and result column name
        'average_duration': 'mean',
        'max_duration': 'max',
        'num_calls': 'count'
    },
    'date': {
        #'max_date': 'max',
        #'min_date': 'min',
        'num_days': lambda x: (max(x) - min(x)).days
    },
    'network': ['count', 'max']
}
data[data['item']=='call'].groupby('month').agg(aggregation)
#               duration                                             date network
#         total_duration average_duration max_duration num_calls num_days   count        max
# month
# 2014-11        25547.0       238.757009       1940.0       107       28     107  voicemail
# 2014-12        13561.0       171.658228       2120.0        79       30      79  voicemail
# 2015-01        17070.0       193.977273       1859.0        88       30      88  voicemail
# 2015-02        14416.0       215.164179       1863.0        67       25      67  voicemail
# 2015-03        21727.0       462.276596      10528.0        47       19      47  voicemail

# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg({'duration':sum,      # find the sum of the durations for each group
                                     'network_type': "count", # find the number of network type entries
                                     'date': 'first'})    # get the first date per group

# To avoid generate a multi-level header, we can drop the first level header and rename the columns
grouped = data.groupby('month').agg({'duration': ['min', 'max', 'mean']})
grouped.columns = grouped.columns.droplevel(level=0)
grouped = grouped.rename(columns={'min': 'min_duration', 'max': 'max_duration', 'mean': 'mean_duration'})
# Using ravel, and a string join, we can create better names for the columns:
grouped.columns = ['_'.join(x) for x in grouped.columns.ravel()]
# grouped.columns.ravel() is an array of tuples for each column names
# array([('duration', 'min'), ('duration', 'max'), ('duration', 'mean')], dtype=object)

# sort result
grouped.sort_index()
grouped.sort_values('mean_duration', ascending=False)

# group by custom bin
bins = [float('-inf'), 0, 1000, 3000, 5000, 10000, 30000, float('inf')]
data['duration_group'] = pd.cut(data['duration'], bins)
data.groupby('duration_group').count()
data.groupby('duration_group').agg({'num_calls': 'sum'})

```

> Notice:
> If you calculate more than one column of results, your result will be a `Dataframe`.
> For a single column of results, the agg function, by default, will produce a `Series`.

References

1. Pandas `agg()` function is using a similar idea of MongoDB aggregation framework
<http://docs.mongodb.org/manual/applications/aggregation/>

2. Pandas groupby user guide
<http://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html>

### Pandas String Operations

Nearly all Python's built-in string methods are mirrored by a Pandas vectorized string method.

``` python
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
monte.str.lower()
# regular expression
monte.str.extract('([A-Za-z]+)', expand=False)
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
```

Miscellaneous methods

Method          | Description
----------------|------------------------------------------------------------------
get()           | Index each element
slice()         | Slice each element
slice_replace() | Replace slice in each element with passed value
cat()           | Concatenate strings
repeat()        | Repeat values
normalize()     | Return Unicode form of string
pad()           | Add whitespace to left, right, or both sides of strings
wrap()          | Split long strings into lines with length less than a given width
join()          | Join strings in each element of the Series with passed separator
get_dummies()   | extract dummy variables as a dataframe

### Time Series

<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html>

``` python
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

now = datetime.now()
now.year
now.month
now.day
now.hour
now.minute
now.second
now.date()
now.weekday()
now.utcnow()

span = now - datetime(2000, 1, 1, 8, 15)
span.days
span.second
span.total_seconds()

timedelta(12, 3600)
#datetime.timedelta(days=12, seconds=3600)

datetime(2019, 3, 12) + timedelta(3) # add 3 days to today
#datetime.datetime(2019, 3, 15, 0, 0)

# date time to string
str(now)
now.strftime('%Y-%m-%d')
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
now.strftime('%c') # to locale string
locale.setlocale(locale.LC_ALL, 'zh_CHS')
now.strftime('%c')

# parse date time string
datetime.strptime('2011-03-11', '%Y-%m-%d')
# or use dateutil.parser
from dateutil.parser import parse
parse('2011-11-03')
parse('2011-11-03 10:39:22 +0800')
parse('2012-03-01T10:00:00.0000Z')

# generate time range
# signature
pandas.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=_NoDefault.no_default, inclusive=None, **kwargs)
# default freq is 1 day
pd.date_range('2023-01-01', '2023-01-31')
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')
pd.timedelta_range(0, periods=9, freq="2H30T")
```

Datetime format string

<https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>

Directive | Meaning                                             | Example
----------|-----------------------------------------------------|-------------------
%a        | weekday abbreviated                                 | Sun
%A        | weekday full name                                   | Sunday
%w        | weeday as a decimal number, 0 for Sunday, etc.      | 0, .., 6
%d        | day of month with zero padding                      | 01, .., 31
%b        | month abbreviated                                   | Jan
%B        | month full name                                     | January
%m        | month as zero padding decimal number                | 01, ..., 12
%y        | year as 2 digits                                    | 19
%Y        | year as 4 digits                                    | 2019
%H        | Hour (24-hour clock) as zero padding decimal number | 00, .., 23
%I        | Hour (12-hour clock) as zero padding decimal number | 01, .., 12
%p        | AM/PM                                               | AM, PM
%M        | Minute as zero padding decimal number               | 00, .., 59
%S        | Second as zero padding decimal number               | 00, .., 59
%f        | Microsecond as zero padding decimal number          | 000000, .., 999999
%z        | UTC offset                                          | +0000, -0800
%Z        | Timezone name                                       | +0000, -0800

``` python
#pd.date_range(datetime.now(), periods=10, freq='10S')
dates = pd.date_range(start=datetime(2019, 1, 1), end=datetime(2019, 1, 15), freq='D').tolist()
ts = pd.Series(np.random.randn(15), index=dates)
ts['2019-01-11']
ts['2019-01-11':'2019-01-15']
ts['2019-01']
ts.truncate(after='2019-01-11')
```

See <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases> for a full `freq` alias list.

Common freq alias

alias | meaning
------|-------------
D     | calendar day
W     | week
M     | month
H     | hour
T,min | minutes
S     | seconds
L,ms  | milliseconds

``` python
# example to load time series data and visualize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from sklearn import datasets, linear_model

data = pd.read_csv('dataset/boost.csv')

data.columns = ['positive_boost', 'no_boost', 'negative_boost', 'date']
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

idx = pd.date_range('2019-01-01', '2019-03-11')
df = data.reindex(idx)
df['offset'] = list((idx - datetime(2019, 1, 1)).days)
df['total'] = df['positive_boost'] + df['no_boost'] + df['negative_boost']
# ['2019-02-17':'2019-02-27'] offset [47:57] are missing

# user linear regression to predict total
Y = df['total'].values.reshape(-1,1)
X = df['offset'].values.reshape(-1,1)

X_train = X[:47]
Y_train = Y[:47]

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

pred_total = regr.predict(X).ravel()
df['pred_total'] = pred_total

# user linear regression to predict positive boost
Y = df['positive_boost'].values.reshape(-1,1)
X = df['offset'].values.reshape(-1,1)

X_train = X[:47]
Y_train = Y[:47]

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

pred_pos = regr.predict(X).ravel()
df['pred_pos'] = pred_pos

# plot bar chart
fig = plt.figure()
ax = plt.subplot(111)

colors = ['#e0eee0', '#bebebe', '#eb9b9b']
df[['positive_boost', 'no_boost', 'negative_boost']].plot(ax=ax, kind='bar', stacked=True, color=colors)

xtl = [item.get_text()[2:10] for item in fig.get_xticklabels()]
_ = fig.set_xticklabels(xtl)

ytl = ['{:,.0f}'.format(item/1000000) + 'M' for item in fig.get_yticks()]
_ = fig.set_yticklabels(ytl)

df[['pred_total', 'pred_pos']].plot(ax=ax)

plt.show()
```

### Eval

The `eval()` function in Pandas uses string expressions to efficiently compute operations using DataFrames.

``` python
# eval DataFrame operations
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3)))
                           for i in range(5))
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)
# result: True

# eval column wise operations
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
df.eval('D = (A - B) / C', inplace=True)

# query
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
result3 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)
```

## Visualization

### Matplotlib

<https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>

``` python
import numpy as np
from numpy.random import randn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# in IPython, run magic command to set backend so that no need to run plt.show()
%matplotlib

# in Jupyter, set backend as inline
%matplotlib inline

x = np.arange(0, 360)
y = np.sin( x * np.pi / 180.0)

fig = plt.figure()
plt.plot(x, y)
plt.xlim(0, 360)        # set x axis range
plt.ylim(-1.0, 1.0)     # set y axis range
plt.title("y = sin(x)") # set title
fig.savefig('sin.png')  # supported formats: png, jpg, svg, pdf, tif, etc.

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)  # create subplot
ax2 = fig.add_subplot(2, 2, 2)
_ = ax1.hist(randn(100), bins=20, color='b', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30)+3*randn(30))
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(randn(50).cumsum(), 'k--')

# set axis ticks & labels
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k--', linewidth=0.5, label='one')
ax.plot(randn(1000).cumsum(), 'b-', linewidth=0.3, label='two')
ax.plot(randn(1000).cumsum(), 'g:', linewidth=1, label='three')
ticks = ax.set_xticks(list(range(0, 1001, 250))) # [0, 250, 500, 750, 1000]
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('Random Walker')
ax.set_xlabel('Stages')
ax.legend(loc='best')
plt.savefig('sample.svg')
plt.savefig('sample.png', dpi=400, bbox_inches='tight')
```

#### Line Styles & Markers

character|description
---------|----------------------
'-'      |solid line style
'--'     |dashed line style
'-.'     |dash-dot line style
':'      |dotted line style
'.'      |point marker
','      |pixel marker
'o'      |circle marker
'v'      |triangle_down marker
'^'      |triangle_up marker
'<'      |triangle_left marker
'>'      |triangle_right marker
'1'      |tri_down marker
'2'      |tri_up marker
'3'      |tri_left marker
'4'      |tri_right marker
's'      |square marker
'p'      |pentagon marker
'*'      |star marker
'h'      |hexagon1 marker
'H'      |hexagon2 marker
'+'      |plus marker
'x'      |x marker
'D'      |diamond marker
'd'      |thin_diamond marker
'\|'     |vline marker
'_'      |hline marker

#### Colors

character|color
---------|--------
'b'      |blue
'g'      |green
'r'      |red
'c'      |cyan
'm'      |magenta
'y'      |yellow
'k'      |black
'w'      |white

#### Plot properties

Property          | Description
------------------|------------------------------------------------------------------------------------------------
alpha             | float (0.0 transparent through 1.0 opaque)
antialiased or aa | True/False
color or c        | any matplotlib color
dashes            | sequence of on/off ink in points
figure            | a Figure instance
fillstyle         | 'full'/'left'/'right'/'bottom'/'top'/'none'
label             | object
linestyle or ls   | 'solid'/'dashed', 'dashdot', 'dotted'/(offset, on-off-dash-seq)/'-'/'--'/'-.'/':'/'None'/' '/''
linewidth or lw   | float value in points
marker            | A valid marker style
markersize or ms  | float
xdata             | 1D array
ydata             | 1D array

<https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>

#### Common pyplot functions

plot Function Name| ax Function Name | Description
-|-|-
xlim, ylim|set_xlim, set_ylim|Set axes limits
axis|-|Customize axis
xlabel, ylabel|set_xlabel, set_ylabel|Set axes label
title|set_title|Plot title
legend|-|Show legend
colorbar|-|Color legend
errorbar, fill-between||Show error in bar or area
vlines, hlines|Set vertical or horizontal guide lines
ax.text|Annotation text
plt.arrow,plt.annotate|Annotation line

#### Common graphics types

Function|Descriptions
-|-
plt.plot, plt.scatter|Scatter plot
plt.bar, plt.barh|Bar plot
plt.pie|notorious pie chart
plt.stackplot|stack area plot
plt.boxplot|With error boxes
plt.violinplot|Centered bar chart
plt.hist|Histogram
np.histogram2d+plt.hist2d+plt.colorbar|Show 3D data in 2D using color as 3rd axis
plt.hexbin+plt.colorbar|3rd dimension in color
plt.imshow+plt.colorbar|3rd dimension in color
np.meshgrid+plt.contour+plt.colorbar|3rd dimension in color

Examples:
<https://matplotlib.org/stable/gallery/images_contours_and_fields/index.html>

``` python
# Show California cities, geolocation as X and Y, city area as dor size, population in color
cities = pd.read_csv('data/california_cities.csv')

# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but no label
plt.scatter(lon, lat, label=None,
            c=np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend:
# we'll plot empty lists with the desired size and label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population');
```

#### Set style

``` python
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')

# render histogram and line with custom style
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
```

Style list:

- fivethirtyeight
- ggplot
- bmh
- dark_background
- grayscle

Using Seaborn

``` python
import seaborn as sns
sns.set()

sns.kdeplot(data, shade=True)
sns.distplot(data['x'])
sns.distplot(data['y'])

with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')

# Visualizing the multidimensional relationships among the samples
sns.pairplot(iris, hue='species', size=2.5)

# Joint plot with automatic kernel density estimation and regression
sns.jointplot("total_bill", "tip", data=tips, kind='reg');

# Time series in bars
with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
```

#### 3D plot

``` python
ax.plot3D          # 3D line
ax.scatter3D       # 3D scatter points
ax.contour3D       # 3D mesh
ax.plot_wireframe  # 3D wireframe
ax.plot_surface    # 3D surface
ax.view_init     # adjust view pont
# Show geographical data using BaseMap module
from mpl_toolkits.basemap import Basemap
```

Matplot Gallery
<http://matplotlib.org/gallery.html>

Matplot figure
<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html>

Axis
<https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes>

Pandas DataFrame plot parameters
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html>

Examples
<https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>

### Pandas plot

df = pd.DataFrame()

``` python
import pandas as pd
df = pd.DataFrame

# build DataFrame from rows
sales = [('Jones LLC', 150, 200, 50),
('Alpha Co', 200, 210, 90),
('Blue Inc', 140, 215, 95)]
labels = ['account', 'Jan', 'Feb', 'Mar']
df = pd.DataFrame.from_records(sales, columns=labels)

df.plot.<TAB>
df.plot.area     df.plot.barh     df.plot.density  df.plot.hist     df.plot.line     df.plot.scatter
df.plot.bar      df.plot.box      df.plot.hexbin   df.plot.kde      df.plot.pie
```

> Reference:
>
> - <https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>
> - <https://plot.ly/pandas/>

### Bokeh

<https://docs.bokeh.org/en/latest/docs/gallery.html>
<https://blog.csdn.net/weixin_38037405/article/details/121498202>

``` python
!pip install pandas-Bokeh

import pandas as pd
from bokeh.plotting import figure
from bokeh.io import show

# is_masc is a one-hot encoded dataframe of responses to the question:
# "Do you identify as masculine?"

#Dataframe Prep
counts = is_masc.sum()
resps = is_masc.columns

#Bokeh
p2 = figure(title='Do You View Yourself As Masculine?',
          x_axis_label='Response',
          y_axis_label='Count',
          x_range=list(resps))
p2.vbar(x=resps, top=counts, width=0.6, fill_color='red', line_color='black')
show(p2)

#Pandas
counts.plot(kind='bar')
```

### Plotly

<https://plotly.com/python/basic-charts/>
<https://www.geeksforgeeks.org/python-plotly-tutorial/>
<https://plot.ly/create/>

``` python
# install plotly
!pip install plotly --upgrade

# import module
import plotly

import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go

# offline plot
offline.plot([{'x': [1, 3, 6],
               'y': [3, 1, 5]}])
offline.iplot([{'x': [1, 3, 6],
               'y': [3, 1, 5]}])
offline.iplot([{'x': [1, 3, 6]}])
offline.iplot([{'y': [3, 1, 5]}])

# use plotly express
import plotly.express as px

# Creating the Line Figure instance
fig = px.line(x=[1, 2, 3], y=[1, 2, 3])
fig.show()

# Bar
fig = px.bar(df, x="sepal_width", y="sepal_length")
# Histogram
fig = px.histogram(df, x="sepal_length", y="petal_width")
# Scatter
fig = px.scatter(df, x="species", y="petal_width")
# Pie
fig = px.pie(df, values="total_bill", names="day")
# plotting the box chart
fig = px.box(df, x="day", y="total_bill")
# plotting the violin chart
fig = px.violin(df, x="day", y="total_bill")
# Gantt
fig = ff.create_gantt(df)
```

## Machine Learning

Categories

- Supervised learning: Models that can predict labels based on labeled training data
  - Classification: Models that predict labels as two or more discrete categories
  - Regression: Models that predict continuous labels
- Unsupervised learning: Models that identify structure in unlabeled data
  - Clustering: Models that detect and identify distinct groups in the data
  - Dimensionality reduction: Models that detect and identify lower-dimensional  structure in higher-dimensional data
- Semi-supervised learning

Common Estimator Training Steps

1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
2. Choose model **hyperparameters** by instantiating this class with desired values.
3. Arrange data into a **features matrix** and **target vector** following the discussion above.
4. Fit the model to your data by calling the `fit()` method of the model instance.
5. Apply the Model to new data:
    - For supervised learning, often we predict labels for unknown data using the predict() method.
    - For unsupervised learning, we often transform or infer properties of the data using the transform() or predict() method.

### Scikit-Learn

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib

# Linear Regression
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
# choose model
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]  # reshape feature matrix
# fit
model.fit(X, y)
model.coef_           # array([1.94667186])
model.intercept_      # -0.5553164627339857
# apply model to new data
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)

# Classification
iris = pd.read_csv('data/iris.csv')
iris.head(5)
sns.pairplot(iris, hue='species', height=1.5)
X_iris = iris.drop('species', axis=1)   # X_iris.shape (150, 4)
y_iris = iris['species']                # y_iris.shape (150,)
# split training set and testing set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
# fit model
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
# evaluate
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

# Unsupervised Learning: reducing dimensionality
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)

# Unsupervised Learning: Clustering
from sklearn.mixture import GaussianMixture   # 1. Choose the model class
model = GaussianMixture(n_components=3, covariance_type='full')  # 2. Instantiate model
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False)

# dimension reducing using IsOmap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

# use confussion matrix to debug model prediction errors
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

# show mislabeled images
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
test_images = Xtest.reshape(-1, 8, 8)
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')
```

References:

- <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
