# Python Data Science Handbook Reading Notes

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

* `In` and `Out` are built-in object arrays for history
* Previously output could be accessed using `_`, `__`, `___`

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
np.random.random((3, 4))
np.random.normal(0, 1, (3, 4))
np.random.randint(0, 10, (3, 4))
np.empty(3)           # create an uninitialized array
# slice sub array
x2 = x[:, ::2]        # all rows, every other column
x3 = x[::-1, ::-1]    # start:end:step, upside-down and reversed in every row
# reshape
grid = np.arange(1, 10).reshape((3, 3))
```

> Notice that NumPy array slice DONT copy the array, it is just a sub-view. If you modify an element value, the value in slice will also change. If you want to get a copy of the data slice, you can use `copy()` method. Modifing the copied data won't change original data.

### UFuncs

Since Python loops could be very slow due to dynamic type check and dispatching, it is recommended to **use UFuncs as much as possible**.

UFuncs:

* Array arithmetic: `+(add), -(subtract), -(negative), *(multiply), /(divide), **(power), //(floor_divide), % (mod), abs`
* Trigonometric functions: `sin, cos, tan, arcsin, arccos, arctan`
* Exponents and logarithms: `exp(e^x), exp2(2^x), power, log = ln(x), log2, log10, expm1 = exp(x)-1, log1p = log(1+x)`
* Hyperbolic trig functions
* Bitwise arithmetic: `logical_and, logical_or, logical_xor, &, |, ^`
* Comparison operators: `greater, greater_equal, less, less_equal, equal, not_equal, >, >=, <, <=, ==, !=`
* Conversions from radians to degrees
* Rounding and remainders
* Other functions defined in scipy.special, ref: <https://docs.scipy.org>
* Aggregates: `reduce, accumulate, sum, min, max, mean, prod, std, var, any, all`

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
data = pd.read_csv('president_heights.csv')
data.head(5)
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

* sum, prod
* min, max: optional parameter to specify which dimension, e.g. x.min(axis=0)
* argmin, argmax: index of min/max
* mean, std, var, median, percentile
* any, all: evaluate whether any/all elements are true
* unique(x), intersect1d(x, y), union1d(x, y), in1d(x, y), setdiff1d(x, y), setxor1d(x, y): basic set operations

### Broadcasting

When 2 array of different dimesion are in same arithmetic, NumPy is trying to extend the array size so that they could match the same bigger size. Rules are:

1. If 2 arrays differ in their number of dimensions, the shape of the one with fewer dimensions is **padded** with ones on its leading (left) side;
2. If the shape of 2 arrays do not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape;
3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

``` python
# example: use broadcasting to compute the z = f(x, y) across the grid
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0,5,0,5])
plt.show()

#example: use broadcasting to normalize data
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

# count with query
np.count_nonzero(x < 6)
np.sum((x > 5) & (x < 10))
np.any(x > 10)
np.all(x > 0, axis=1)
```

## Visualization

### Plotly

* Business Intelligence

  * Chart Studio
  * Dashboards & Reporting
  * Slide Decks

* Data Science & Open Source

  * Dash
  * Plotly.py, Plotly.R, Plotly.js
  * React

* Platforms

  * Plotly On-premise
  * Plotly Cloud

Data -> Layout -> Figure

Offline Plots

``` python
# install plotly
!pip install plotly --upgrade

# import module
import plotly
plotly.__version__

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
```

<https://plot.ly/create/>

Online Plots

Sankey plot

Matplotlib => Seaborn

### Matplotlib

``` python
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

x = np.arange(0, 360)
y = np.sin( x * np.pi / 180.0)

plt.plot(x, y)
plt.xlim(0, 360)        # set x axis range
plt.ylim(-1.0, 1.0)     # set y axis range
plt.title("y = sin(x)") # set title
plt.show()

# create an empty figure and add subplots
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
_ = ax1.hist(randn(100), bins=20, color='b', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30)+3*randn(30))
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(randn(50).cumsum(), 'k--')
plt.show()

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
fig.show()
plt.savefig('sample.svg')
plt.savefig('sample.png', dpi=400, bbox_inches='tight')
```

#### Line Styles & Markers

character/description
----------|----------------------
'-'      /solid line style
'--'     /dashed line style
'-.'     /dash-dot line style
':'      /dotted line style
'.'      /point marker
','      /pixel marker
'o'      /circle marker
'v'      /triangle_down marker
'^'      /triangle_up marker
'<'      /triangle_left marker
'>'      /triangle_right marker
'1'      /tri_down marker
'2'      /tri_up marker
'3'      /tri_left marker
'4'      /tri_right marker
's'      /square marker
'p'      /pentagon marker
'*'      /star marker
'h'      /hexagon1 marker
'H'      /hexagon2 marker
'+'      /plus marker
'x'      /x marker
'D'      /diamond marker
'd'      /thin_diamond marker
'\|'     /vline marker
'_'      /hline marker

#### Colors

character/color
----------|--------
'b'      /blue
'g'      /green
'r'      /red
'c'      /cyan
'm'      /magenta
'y'      /yellow
'k'      /black
'w'      /white

#### plot properties

Property|Description
-|-
alpha|float (0.0 transparent through 1.0 opaque)
antialiased or aa|True/False
color or c|any matplotlib color
dashes|sequence of on/off ink in points
figure|a Figure instance
fillstyle|'full'/'left'/'right'/'bottom'/'top'/'none'
label|object
linestyle or ls|'solid'/'dashed', 'dashdot', 'dotted'/(offset, on-off-dash-seq)/'-'/'--'/'-.'/':'/'None'/' '/''
linewidth or lw|float value in points
marker|A valid marker style
markersize or ms|float
xdata|1D array
ydata|1D array

> Reference:
> <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>

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
> * <https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>
> * <https://plot.ly/pandas/>

## Data Aggregation

``` python
import numpy as np
import pandas as pd

people = pd.DataFrame(np.random.randn(5, 5),
    columns=['a', 'b', 'c', 'd', 'e'],
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
mapping = {'a':'red', 'b':'red', 'c':'blue', 'd':'blue', 'e':'red'}

people.groupby(mapping, axis=1).sum()
grouped = people.groupby(mapping, axis=1)
grouped.describe()
```

### Pivot table and crosstab

``` python
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value=0)
# C        large  small
# A   B
# bar one    4.0    5.0
#     two    7.0    6.0
# foo one    4.0    1.0
#     two    NaN    6.0

table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean, 'E': [min, max, np.mean]})

# The definition of pandas.pivot_table function is:
pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All')[source]

# crosstab is a easier vresion of pivot_table
pd.crosstab(df.A, df.B, margins=True)

# B    one  two  All
# A
# bar    2    2    4
# foo    3    2    5
# All    5    4    9

pd.crosstab([df.A, df.B], df.C, margins=True)

# C        large  small  All
# A   B
# bar one      1      1    2
#     two      1      1    2
# foo one      2      1    3
#     two      0      2    2
# All          4      5    9

```

Parameters

* `data` : DataFrame
* `values` : column to aggregate, optional
* `index` : column, Grouper, array, or list of the previous
* `columns` : column, Grouper, array, or list of the previous
* `aggfunc` : function, list of functions, dict, default numpy.mean
* `fill_value` : scalar, default None, Value to replace missing values with
* `margins` : boolean, default False, Add all row / columns (e.g. for subtotal / grand totals)
* `dropna` : boolean, default True, Do not include columns whose entries are all NaN
* `margins_name` : string, default 'All', Name of the row / column that will contain the totals when margins is True.

#### A Full Example of Data Aggregation and Grouping in Pandas

Reference:
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

### Reference

1. Pandas `agg()` function is using a similar idea of MongoDB aggregation framework
<http://docs.mongodb.org/manual/applications/aggregation/>

2. Pandas groupby user guide
<http://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html>

## Time Series

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
```

### Datetime format string

<https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>

Directive|Meaning|Example
-|-|-
%a|weekday abbreviated|Sun
%A|weekday full name|Sunday
%w|weeday as a decimal number, 0 for Sunday, etc.|0, .., 6
%d|day of month with zero padding|01, .., 31
%b|month abbreviated|Jan
%B|month full name|January
%m|month as zero padding decimal number|01, ..., 12
%y|year as 2 digits|19
%Y|year as 4 digits|2019
%H|Hour (24-hour clock) as zero padding decimal number|00, .., 23
%I|Hour (12-hour clock) as zero padding decimal number|01, .., 12
%p|AM/PM|AM, PM
%M|Minute as zero padding decimal number|00, .., 59
%S|Second as zero padding decimal number|00, .., 59
%f|Microsecond as zero padding decimal number|000000, .., 999999
%z|UTC offset|+0000, -0800
%Z|Timezone name|+0000, -0800

### Pandas TimeSeries

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

alias|meaning
-|-
D|calendar day
W|week
M|month
H|hour
T,min|minutes
S|seconds
L,ms|milliseconds

``` python
# example to load time series data
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

Matplot figure
<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html>

Axis
<https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes>

Pandas DataFrame plot parameters
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html>

Examples
<https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html>

## Combining Datasets

Common functions in Pandas to combine DataFrames are:

* concat()
* append()
* merge()
* join()

``` python
# join 2 DataFrame on specified key
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
```
