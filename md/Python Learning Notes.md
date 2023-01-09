# Python Learning Notes

- [Python Learning Notes](#python-learning-notes)
  - [Basics](#basics)
    - [Comments](#comments)
    - [Basic Types](#basic-types)
    - [Operators](#operators)
    - [Bitwise operations](#bitwise-operations)
  - [String Operations](#string-operations)
    - [String Formatting](#string-formatting)
    - [String functions](#string-functions)
    - [Get substring](#get-substring)
    - [Loop through string](#loop-through-string)
  - [List operations](#list-operations)
    - [Range](#range)
    - [List comprehension](#list-comprehension)
    - [List Slicing](#list-slicing)
    - [enumerate](#enumerate)
    - [find unique elements](#find-unique-elements)
  - [Built-in Data Types](#built-in-data-types)
    - [Dictionary](#dictionary)
    - [Tuples](#tuples)
    - [Set](#set)
    - [Random](#random)
    - [Inspect an object](#inspect-an-object)
  - [Control Flow](#control-flow)
    - [Condition](#condition)
    - [Try/except](#tryexcept)
  - [Functions](#functions)
    - [Define function](#define-function)
    - [Optional parameter](#optional-parameter)
    - [Variable parameter](#variable-parameter)
    - [Keyword parameter](#keyword-parameter)
    - [Anonymous function](#anonymous-function)
    - [Unit testing](#unit-testing)
    - [Local variable and global variable](#local-variable-and-global-variable)
    - [Shallow copy / deep copy](#shallow-copy--deep-copy)
    - [Functional programming](#functional-programming)
  - [Class syntax](#class-syntax)
    - [Inherit class](#inherit-class)
    - [Properties](#properties)
    - [Exceptions](#exceptions)
    - [Logging](#logging)
  - [I/O](#io)
    - [Gather user input](#gather-user-input)
    - [Import library](#import-library)
    - [Reading \& Writing Files](#reading--writing-files)
    - [Read write binary files](#read-write-binary-files)
    - [Read console input](#read-console-input)
    - [Read/Write Clipboard](#readwrite-clipboard)
    - [Read data from a JSON file](#read-data-from-a-json-file)
    - [Read data from .dat file](#read-data-from-dat-file)
    - [Web Scraping](#web-scraping)
    - [Automate Web Browser](#automate-web-browser)
      - [1. Install Selenium](#1-install-selenium)
      - [2. Install Web Drivers](#2-install-web-drivers)
      - [3. Start browser, find element, and send keys](#3-start-browser-find-element-and-send-keys)
      - [Find elements](#find-elements)
      - [Common expected conditions](#common-expected-conditions)
  - [Regular Expression](#regular-expression)

## Basics

### Comments

Single line comment: start with #

``` python
# This is a line of comment
```

Multiple line comment: surrounded by triple quotes

``` python
"""
Multiple line comment: surrounded by triple quotes
"""
```

### Basic Types

- Integers
- Float
- String (surrounded by '' or "")
- Boolean: `True` `False`

> TIPS: You can look up a variable's type using function `type()`

### Operators

- Arithmetic: `+` `-` `*` `/`  `%` (mod) `**` (exponent)
- String:
- Logic: `and` `or` `not` (priority: `not` > `and` > `or`)
- Comparison: `==`  `!=`  `>`  `<`  `>=`  `<=`
- Bitwise operation: `>>`  `<<` (shift left/right) `&`  `|`  `^`  `~`  (and/or/xor/not)

---

### Bitwise operations

``` python
print 5 >> 4  # Right Shift
print 5 << 1  # Left Shift
print 8 & 5   # Bitwise AND
print 9 | 4   # Bitwise OR
print 12 ^ 42 # Bitwise XOR
print ~88     # Bitwise NOT
```

---

## String Operations

### String Formatting

The `%` operatpr after a string is used to combine a string with variables. The `%` operator will replace a `%s` in the string with the string variable that comes after it.
An easier way is to use string interporation, starting with `f`, then use the variable names in `{}`.

Example:

``` python
name_1 = "Mike"
name_2 = "Dean"
print "Hello %s and %s" % (name_1, name_2)
print(f"Hello {name_1} and {name_2}")
```

Parse sting to integer using built-in function

``` python
num = int('10')
```

### String functions

``` python
.len()     # get length
.isalpha() # is alphabetic
.lower()   # is in lower case
```

### Get substring

- `str[index]` gets the index character in str
- `str[start:end]` gets the substring in str (including start, not include end)

### Loop through string

``` python
for c in str:
    print c
```

---

## List operations

``` python
array = [1, 2, 5, False, 'string']
len(array)              # gets the length of list
array.append(99)        # add new element to the end of list
array.insert(1, 'new')  # insert 'new' at index 1
slice = array[1:3]      # get sub list from index 1 to 2
first_two = array[:2]   # get first 2 elements
from_3 = array[2:]      # get substring from 3rd element
del array[:1]           # delete first element in list
array.remove('string')  # remove certain element from list
r = array.pop(1)        # remove and return element at index

animals = ["cat", "dog", "bat"]
animals.sort()          # sort the list
animals.index["bat"]    # get index of an element
for a in animals:       # loop by element
    print(a)
for index in range(len(animals)): # loop by index
    print("index = ", index, ", element = ", animals[index])
", ".join(animals)      # join string array to generate "cat, dog, bat"

# keyword in could also be used to find an item in an array
animals = ["cat", "dog", "rabbit"]
if "cat" in animals:
    print("Cat found")
```

### Range

Notice that when using `for..in` syntax (for item in list), the item is readonly, if you change the value of item, the changed value will not be saved in list.
To update items in list, you need to use `range()` function to get the indexes.

Example:

``` python
list = [i for i in range(10)]
for i in range(0, len(list)):
    list[i] *= 2
```

Range function generates a list of numbers

`range(start, stop, step)`

`start` is inclusive and `stop` is exclusive
by default `start` is 0 and `stop` is length of array, `step` is 1

2 lists could be joined together using `+`

``` python
m = [1, 2, 3]
n = [4, 5]
print m + n     # [1, 2, 3, 4, 5]
```

Check number in range

``` python
if n not in range(10) or \
    m not in range(6):
        print "number not in range"
```

### List comprehension

``` python
evens_to_50 = [i for i in range(51) if i % 2 == 0]
doubles_by_3 = [x*2 for x in range(1,6) if (x*2)%3 == 0]
# => [6]

# comprehension could alse be used in dict
# dict of the squares from 1^2 to 10^2, where the keys are the unsquared numbers
squares_dict = {i: i**2 for i in range(1, 11)}
```

### List Slicing

List slicing allows us to access elements of a list in a concise manner

`[start:end:stride]`

`start` is inclusive and `end` is exclusive
stride could be negative, for example, following code return a list in reverse

`list[::-1]`

### enumerate

``` python
for idx, value in enumerate(['foo', 'bar']):
    print(idx, value)
```

### find unique elements

The function `set()` returns only the unique elements in a list

``` python
weeks = [1,1,1,2,2,2,3]
unique_weeks = set(weeks)  # {1,2,3}
```

---

## Built-in Data Types

### Dictionary

``` python
dict = {'key1': 0, 'key2': 1}
dict['key1']            # get value of a key
dict['new_key'] = 90    # add a new key/value pair
del dict['new_key']     # delete a key/value pair
for key in dict:
    print dict[key]
print dict.items()
print dict.keys()
print dict.values()
print 'key' in dict     # check if a key is in the dictionary
```

### Tuples

Tuples are immutable collections of items that are surrounded by `()`, the items could be any data type

``` python
a_tuple = (12, 3, 5, 15, 6)
another_tuple = 12, 3, 5, 15, 6

# generate a tuple list by zip 2 or more lists
a = [1,2,3]
b = [4,5,6]
ab = zip(a, b)

print(list(ab))
for i,j in zip(a,b):
    print(i, j)
```

Named tuples could be used similar as class.

``` python
import collections

people = [("Michele", "Vallisneri", "July 15"),
          ("Albert", "Einstein", "March 14"),
          ("John", "Lennon", "October 9"),
          ("Jocelyn", "Bell Burnell", "July 15")]
# declare a named tuple as type
persontype = collections.namedtuple('person', ['firstname', 'lastname', "birthday"])
# create instance using names or by order
michele = persontype("Michele", "Vallisneri", "July 15")
michele = persontype(lastname="Vallisneri", firstname="Michele", birthday="July 15")
michele[0], michele[1], michele[2]
michele.firstname, michele.lastname, michele.birthday
# use tuple unpacking on people[0] to build a namedtuple
persontype(*people[0])
namedpeople = [persontype(*person) for person in people]

# !pip install dataclasses
from dataclasses import dataclass
# defining a data class with the same content as the "person" nametuple
# and with a default for "birthday"

@dataclass
class personclass:
    firstname: str
    lastname: str
    birthday: str = 'unknown'
    # all methods in a class carry a conventional argument "self";
    # when the methods are called on an instance (here, a specific person),
    # "self" points the instance itself, so self.firstname and self.lastname
    # are the data fields in that instance
    def fullname(self):
        return self.firstname + ' ' + self.lastname

michele = personclass(firstname='Michele', lastname='Vallisneri')

```

---

### Set

``` python
char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']
sentence = 'Welcome Back to This Tutorial'

print(set(char_list))
# {'b', 'd', 'a', 'c'}

print(set(sentence))
# {'l', 'm', 'a', 'c', 't', 'r', 's', ' ', 'o', 'W', 'T', 'B', 'i', 'e', 'u', 'h', 'k'}

print(set(char_list + list(sentence)))
# {'l', 'm', 'a', 'c', 't', 'r', 's', ' ', 'd', 'o', 'W', 'T', 'B', 'i', 'e', 'k', 'h', 'u', 'b'}

unique_char = set(char_list)
unique_char.add('x')
# unique_char.add(['y', 'z']) this is wrong
print(unique_char)
# {'x', 'b', 'd', 'c', 'a'}

unique_char.remove('x')
print(unique_char)
# {'b', 'd', 'c', 'a'}

unique_char.discard('d')
print(unique_char)
# {'b', 'c', 'a'}

unique_char.clear()
print(unique_char)
# set()

unique_char = set(char_list)
print(unique_char.difference({'a', 'e', 'i'}))
# {'b', 'd', 'c'}

print(unique_char.intersection({'a', 'e', 'i'}))
# {'a'}

print(unique_char.union({'a', 'e', 'i'}))
# {'a', 'b', 'c', 'd', 'e', 'i'}
```

---

### Random

``` python
from random import randint
n = ranint(0, 10)      # random integer between [0, 10]
```

### Inspect an object

``` python
# check what variables or functions available in IPython or Jupyter
%who

# get object type
obj = {'a': 1, 'b': 124.5, 'c': True, 'd': 'some value'}
type(obj)

# get properties and functions
from inspect import getmembers
getmembers(obj)

# stringify
import json
json.dumps(obj)
```

---

## Control Flow

### Condition

``` python
if option == '1' or option == '2':
    print 'option 1 or 2'
elif option == '3':
    print 'option 3'
else:
    print 'other options'
```

---

### Try/except

``` python
try:
    file = open('eeee.txt','r')
except Exception as e:
    print(e)
```

---

## Functions

### Define function

Notice that indent is critical

``` python
def func():
    """docString for this function, could be access via func? """
    print "do something"
    return 0

# call it
func()
```

Return statement is optional

### Optional parameter

``` python
# Optional parameter can only be after normal parameters.
def counter(list, hasHeader = False):
    count = 0
    if hasHeader: # if list has header, remove the 1st record
        list = list[1:leng(list)]
    for record in list:
        count = count + 1
    return count
```

### Variable parameter

``` python
# variable parameter can only be after normal parameters and optional parameters
def report(name, *grades):
    total_grade = 0
    for grade in grades:
        total_grade += grade
    print(name, 'total grade is ', total_grade)

report('Mike', 8, 9)
report('Mike', 8, 9, 10)
```

### Keyword parameter

``` python
# keyword parameter can only be at last of parameters
def portrait(name, **kw):
    print('name is', name)
    for k,v in kw.items():
        print(k, v)

portrait('Mike', age=24, country='China', education='bachelor')
```

Any function could be described as `universal_func(*args, **kw)`

### Anonymous function

define a lambda expression

``` python
lambda x: x % 3 == 0
```

is the same as

``` python
def by_three(x):
    return x % 3 == 0
```

Example:

``` python
list = range(16)
result = filter(lambda x: x%3 == 0, list)
# result: [0,3,6,9,12,15]

# Use lambda function to define map
list(map(lambda x, y: x + y, [1, 2], [3, 4]))
```

### Unit testing

``` python

if __name__ == '__main__':
    # unit test code here
    # they won't be executed when script is imported as module
```

### Local variable and global variable

- **Local variable**: variable defined in a function
- **Global variable**: variable defined outside function and could be called in function with keyword `global`

``` python
a = None
def fun():
    global a
    a = 20
    return a + 100
```

### Shallow copy / deep copy

``` python
a = [1, 2, 3]
id(a)                  # memory address of a
id(a) == id(a[0])      # False

# Shallow copy
b = a
# Deep copy
import copy
c = copy.copy(a)
a[1] = -1

id(a[1]) == id(b[1])   # True
id(a[1]) == id(c[1])   # False
id(a[0]) == id(b[0])   # True
id(a[0]) == id(c[0])   # True, python is smart enough not to duplicate yet another memory
```

### Functional programming

Python allows anonymous function and passing functions as parameters, which provides convenience for functional programming.

---

## Class syntax

``` python
# class inherit from object by default
class NewClass(object):  # every class inherits from object class
    # class method always takes at least 1 parameter 'self'
    def __init__(self, name):
        # assign ctor parameter to a member
        # keyword 'self' is like 'this'
        self.name = name
        self.type = 'NewClass'
    def description(self):
        print self.name + self.type

myObject = NewClass('myname')
print myObject.name
print myObject.description
```

### Inherit class

``` python
class Bear(object):
  def __init__(self, name):
    self.name = name

class Panda(Bear):   # class Panda inherits class Bear
  def DisplayName(self):
    print self.name
```

When you override a function definition, you can call the base implementation using **super**

``` python
class Derived(Base):
   def m(self):
       return super(Derived, self).m()
```

Where `m()` is a method from the base class.

When print an object, the object's `__str__()` (string) or `__repr__()` method (representation) is called, so if we'd like to customize a type's default display, we can override this method

``` python
class Dataset:
    def __init__(self, data):
        self.header = data[0]
        self.data = data[1:]

    def __str__(self):
        return str(self.data[:10])
```

### Properties

To access/update value from outside of a class, we could implement properties.

``` python
def class Book:
    def __init__(self, title, author):
        self._title = title
        self._author = author

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

b = Book("War and Peace", "Leo Tolstoy")
print(b.title)
b.title = "Anna Karenina"
print(b.title)

```

### Exceptions

``` python
import traceback

def boxPrint(symbol, width, height):
    if len(symbol) != 1:
        raise Exception('Symbol must be a single character string.')
    if width <= 2 or height <= 2:
        raise Exception('Width and height must be greater than 2.')
    print(symbol * width)
    for i in range(height - 2):
        print(symbol + (' ' * (width - 2)) + symbol)
    print(symbol * width)

def foo(sym, w, h):
    boxPrint(sym, w, h)

def errorLogging(filename):
    errorFile = open(filename, 'w')
    errorFile.write(traceback.format_exc())
    errorFile.close()

for sym, w, h in (('*', 4, 4), ('0', 20, 5), ('x', 1, 3), ('ZZ', 3, 3)):
    try:
        foo(sym, w, h)
    except Exception as err:
        print('An exception happened: ' + str(err))
        errorLogging('errorInfo.txt')
        print('The trackback info was written into errorInfo.txt')
```

### Logging

Reference: [https://docs.python.org/3/howto/logging.html]

``` python
import logging
logging.basicConfig(
    filename='factorial_logs.log',
    filemode='w',
    level=logging.DEBUG,
    force=True,
    format=' %(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')

def factorial(n):
    logging.debug('Start of factorial(%s)' % (n))
    total = 1
    for i in range(1, n + 1):
        total *= i
        logging.debug('i is ' + str(i) + ', total is ' + str(total))
    logging.debug('End of factorial(%s)' % (n))
    return total

print(factorial(5))
logging.debug('End of program')
```

``` python
# shutdown logging when main exit and unlock the logging file
try:
    main()
finally:
    logging.shutdown()
```

## I/O

### Gather user input

``` python
option = raw_input("Type your option and hit 'Enter'.").lower()
```

---

### Import library

import only specified function from a module

`from <module> import <function>`

``` python
from datetime import datetime
now = datetime.now()
# print mm/dd/yyyy
print '%s/%s/%s' % (now.month, now.day, now.year)
#print hh:mm:ss
print '%s:%s:%s' % (now.hour, now.minute, now.second)

import a whole module and reference function by module name
import math
print math.sqrt(25)
print dir(math)  # print all function names from math module

import everything from a module, please DONT using this to avoid naming conflict
from math import *
print sqrt(25)
```

---

### Reading & Writing Files

``` python
# Read a csv and split it to rows and columns
def readDataFile(fileName):
    file = open(fileName, 'r')
    text = file.read()
    lines = text.split('\n')
    rows = []
    for line in lines:
        cols = line.split(',')
        rows.append(cols)
    return rows

# Write to a file
f = open("filename.txt", "w")
f.write("sec")
f.close()

# Another syntax is to use with/as keyword to prevent forgetting to close file
with open("file", "mode") as variable:
    # do something to the file

import os
os.makedirs(r'C:\doc\scripts\test')
pwd = os.getcwd()
newpath = os.path.join(pwd, 'new.txt')

# get all files recursively under a path
root = r'C:\Users\jiew\Documents'
for dirpath, dirnames, filenames in os.walk(root):
    for file in filenames:
        print(os.path.join(os.path.relpath(dirpath, root), file))

# use shutil to copy, rename, move, delete files
import shutil
os.chdir(r'C:\')
shutil.copy('egg.txt', 'target.txt')
shutil.copytree(r'C:\src', r'C:\tgt')

import zipfile
newZip = zipfile.ZipFile('new.zip', 'w')
newZip.write('Basic practice.ipynb', compress_type=zipfile.ZIP_DEFLATED)
newZip.close()

sampleZip = zipfile.ZipFile('new.zip')
sampleZip.extractall(r'C:\somefolder')
sampleZip.close()
```

The zipfile module provides a simple command-line interface to interact with ZIP archives.

If you want to create a new ZIP archive, specify its name after the -c option and then list the filename(s) that should be included:

``` console
python -m zipfile -c monty.zip spam.txt eggs.txt
python -m zipfile -c monty.zip life-of-brian_1979/
```

If you want to extract a ZIP archive into the specified directory, use the -e option:

``` console
python -m zipfile -e monty.zip target-dir/
```

### Read write binary files

``` python
# Using pickle to save/load variables
import pickle
a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}

# pickle a variable to a file
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()

# reload a file to a variable
with open('pickle_example.pickle', 'rb') as file:
    a_dict1 =pickle.load(file)

print(a_dict1)
```

``` python
import shelve
shel = shelve.open('mydata')
cats = ['Zephie', 'Pooks', 'Simon']
shel['cats'] = cats
print(shel['cats'])
print(shel.keys())
shel.close()
```

---

### Read console input

``` python
a_input = int(input('please input a number:'))
if a_input == 1:
    print('This is a good one')
elif a_input == 2:
    print('See you next time')
else:
    print('Good luck')
```

### Read/Write Clipboard

``` python
import win32clipboard                    # installed as part of pywin32 package
win32clipboard.OpenClipboard()
data = win32clipboard.GetClipboardData() # read clipboard data
win32clipboard.SetClipboardText('testing 123')
win32clipboard.EmptyClipboard()
win32clipboard.CloseClipboard()
```

### Read data from a JSON file

`json` module offers functions to handle JSON data, for example, `json.dumps(obj)` to dump object to string.

```python
import json

# load from string
json_string = ''
parsed = json.loads(json_string)
print(json.dumps(parsed, indent=2, sort_keys=True))

# load from a file
with open('tags_post.json', 'r') as handle:
  parsed = json.load(handle)
  f = open('tags_post.formatted.json', 'w')
  f.write(json.dumps(parsed, indent=2, sort_keys=True))
  f.close()
# print(json.dumps(parsed, indent=2, sort_keys=True))

path = '.json'
# read lines from a file into records
records = [json.loads(line) for line in open(path)]
# extract certain field of each record
time_zones = [rec['tz'] for rec in records if 'tz' in rec]

# count words frequency
from collections import defaultdict
def get_counts(sequence):
    counts = defaultdict(int) # initialize every value to 0
    for x in sequence:
        counts[x] += 1
        return counts

# get top n from a dict
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

# count using pandas
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
frame = DataFrame(records)
# drop records with a == null
cframe = frame[frame.a.notnull()]
# slice of a field
frame['tz']
# get a Series from field a's first word
agents = Series([x.split()[0] for x in frame.a.dropna()])
# mapping to an array on a's value
os = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
by_tz_os = cframe.groupby(['tz', os])
agg_counts = by_tz_os.size().unstack().fillna(0)
# get value counts
tz_counts = frame['tz'].value_counts()
# top 10
tz_counts[:10]
# fill missing value of a field
clean_tz = frame['tz'].fillna('Missing') # fill NA with 'Missing'
clean_tz[clean_tz == ''] = 'Unknown'     # fill '' with 'Unknown'
tz_counts = clean_tz.value_counts()
# draw a horizontal bar chart
tz_counts[:10].plot(kind='barh', rot=0)
```

### Read data from .dat file

.dat file each column is separated by ::

movies.dat

``` data
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
```

users.dat

``` data
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
```

``` python
import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('users.dat', sep='::', header=None, names=unames)

# merge tables (automatically by column name)
data = pd.merge(pd.merge(ratings, users), movies)

# pivot table
mean_ratings = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')
ratings_by_title = data.groupby('title').size()

# indexing
active_titles = ratings_by_title.index[ratings_by_title >= 250]
mean_ratings = mean_ratings.ix[active_titles]
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]
```

### Web Scraping

``` python
import requests
res = requests.get(r'https://www.gutenberg.org/files/69432/69432-h/69432-h.htm')

# write response to file
resFile = open(r'The explorer.html', 'wb')
for chunk in res.iter_content(100000);
    resFile.write(chunk)
resFile.close()

import bs4
htmlFile = open('The explorer.html')
soup = bs4.BeautifulSoup(htmlFile.read())
elems = soup.select('h3')
len(elems)
elems[0].getText()
pe = soup.select('p')[0]  # select first <p>
pe.attrs                  # get all attributes
pe.get('styles')          # get styles attribute value
```

### Automate Web Browser

References:

- [https://github.com/SergeyPirogov/webdriver_manager]
- [https://www.selenium.dev/documentation/webdriver/getting_started/first_script/]

#### 1. Install Selenium

``` console
pip install selenium
pip install webdriver-manager
```

#### 2. Install Web Drivers

``` python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
```

#### 3. Start browser, find element, and send keys

``` python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import os

browser = webdriver.Chrome()
browser.delete_all_cookies()
browser.fullscreen_window()
browser.get("https://www.google.com/")

textbox = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, value="input"))
)
#textbox.clear()
#textbox.send_keys("Techbeamers")
#textbox.submit()
browser.find_element(By.NAME, "q").send_keys("Techbeamers" + Keys.ENTER)
wait = WebDriverWait(browser, 10)

browser.save_screenshot('google.png')

# write page source content to file
filename = os.path.join(r'C:\Downloads', 'Page.html')
f = open(filename, 'w')
html = browser.page_source
f.write(html)
f.close()

browser.quit()
```

#### Find elements

``` python
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.example.com")

element = driver.find_element(By.TAG_NAME, 'div')

elements = element.find_elements(By.TAG_NAME, 'p')
for e in elements:
    print(e.text)

driver.find_element(By.CSS_SELECTOR, '[name="q"]').send_keys("webElement")
# Get attribute of current active element
attr = driver.switch_to.active_element.get_attribute("title")

# common selectors
fruits = driver.find_element(By.ID, "fruits")
fruit = fruits.find_element(By.CLASS_NAME,"tomatoes")
fruit = driver.find_element(By.CSS_SELECTOR,"#fruits .tomatoes")
plants = driver.find_elements(By.TAG_NAME, "li")
```

#### Common expected conditions

Ref: [https://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.support.expected_conditions]

- title_is
- title_contains
- presence_of_element_located
- visibility_of_element_located
- visibility_of
- presence_of_all_elements_located
- element_located_to_be_selected
- element_selection_state_to_be
- element_located_selection_state_to_be
- alert_is_present

## Regular Expression

Reference: <https://docs.python.org/3/howto/regex.html>
Regular expression online test tool: <https://www.regexpal.com/>

Meta-characters
`. ^ $ * + ? { } [ ] \ | ( )`

<!-- markdownlint-disable MD030 MD032 -->
Expression | Meaning
-----------|-----------------------------------------
\d         | decimal digit [0-9]
\D         | non-digit character [^0-9]
\s         | whitespace character [ \t\n\r\f\v]
\S         | non-whitespace character [^ \t\n\r\f\v]
\w         | alphanumeric character [a-zA-Z0-9_]
\W         | non-alphanumeric character [^a-zA-Z0-9_]
\b         | blank
\B         | not blank
.          | any character
^          | start
$          | end
\\         | match \
?          | 0 or 1 occurrence
`*`        | 0 or multiple occurrence
`+`        | 1 or multiple occurrence
<!-- markdownlint-enable MD030 MD032 -->

``` python
import re

p = re.compile('ab*', re.IGNORECASE)
m = p.match("abx")

m.group()
m.start(), m.end()
m.span()

iterator = p.finditer('abxababaabbAb')
for match in iterator:
    print(match.span())

# grouping: ?P<group_name>
match = re.search(r"(?P<id>\d+), Date: (?P<date>.+)", "ID: 021523, Date: Feb/12/2017")
print(match.group('id'))                # 021523
print(match.group('date'))              # Date: Feb/12/2017

# Match group with names
m = re.match(r'(?P<first>\w+) (?P<last>\w+)', 'Jane Doe')
print(m.groupdict()) # {'first': 'Jane', 'last': 'Doe'}

# Open file
f = open('test.txt', 'r')
# Feed the file text into findall(); it returns a list of all the found strings
strings = re.findall(r'some pattern', f.read())

## re.sub(pat, replacement, str) -- returns new string with all replacements,
## \1 is group(1), \2 group(2) in the replacement
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
print(re.sub(r'([\w\.-]+)@([\w\.-]+)', r'\1@yo-yo-dyne.com', str))
## purple alice@yo-yo-dyne.com, blah monkey bob@yo-yo-dyne.com blah dishwasher
```

![Regular Expression Cheat Sheet](../images/Regular%20Expressions.png)
