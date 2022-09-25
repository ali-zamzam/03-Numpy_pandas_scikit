"""pandas"""

import numpy as np
import pandas as pd

"""important to study next"""
"""group by and aggregate"""
# df.value_counts()
# df.groupby(df["sex"])["target"].agg(["value_counts"])


"""remove outliers"""
# new_df = df[(df["thalach"] > 50) & (df["thalach"] < 250)]

# new_df["thalach"]

"""change outliers value"""

# df.loc[df["thalach"] > 250, "thalach"] = df['thalach'].max()

# df.loc[df["thalach"] <50, "thalach"] = df['thalach'].min()


"""axis"""
# axis = 0 to change the value in the **column**.
# axis = 1 to change the value in the **row**.
# +------------+---------+--------+
# |            |  A      |  B     |
# +------------+---------+---------
# |      0     | 0.626386| 1.52325|----axis=1----->
# +------------+---------+--------+
#              |         |
#              | axis=0  |
#              ↓         ↓

"""drop column or row"""
#  axis=0 will remove a row instead of a column.
#  If you wish to remove a column, set the axis=1.
# X = heart_disease.drop(["target"], axis=1)

"""with loc"""
# df.drop(df.loc[:, ['Locations', 'Founder']], axis=1)

"""with iloc"""
# df.drop(df.iloc[:,2:9], axis=1)

"""replace an unknown char by NaN"""
# df.replace(r"?", np.nan, inplace=True)

"""change the Nan from multi column to 0"""
# df.update(
#     df[
#         ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
#     ].fillna(0)
"""replace with dic """  # ???? (à voir)

#    Courses    Fee Duration  Discount
# 0    Spark  22000   30days      1000
# 1  PySpark  25000   50days      2300
# 2   Hadoop  23000   30days      1000
# 3   Python  24000     None      1200
# 4   Pandas  26000      NaN      2500
# Now we will remap the values of the 'Courses‘ column by their respective codes using the df.replace() function.


# # Difine Dict with the key-value pair to remap.
# dict = {"Spark" : 'S', "PySpark" : 'P', "Hadoop": 'H', "Python" : 'P', "Pandas": 'P'}
# df2=df.replace({"Courses": dict})
# print(df2)
# Yields below output.


#   Courses    Fee Duration  Discount
# 0       S  22000   30days      1000
# 1       P  25000   50days      2300
# 2       H  23000   30days      1000
# 3       P  24000     None      1200
# 4       P  26000      NaN      2500

"""replace the 0 by the mean"""
# df = df.replace(0, df.mean())

"""Display of the dimensions of my_dataframe"""
# print(df.shape)

"""Display of the dimensions of rows"""
# votes.shape[0]

"""Display of the dimensions of columns"""
# votes.shape[1]
# ----------------------------------------------------------------
"""# Separation of DataFrame"""
# part_1 = transactions[transactions.columns[:4]]
# part_2 = transactions[transactions.columns[4:]]

"""# or """
# part_1 = transactions[["cust_id","tran_date","prod_subcat_code","prod_cat_code"]]
# part_2 = transactions[["qty","rate","tax","total_amt","store_type"]]

"""or use iloc"""
# df1 = datasX.iloc[:, :72]
# df2 = datasX.iloc[:, 72:]
# ---------------------------------------------------------------------------------------------------------

# What is categorical data in Dataframe?
"""A categorical variable takes on a limited, and usually fixed, number of possible values. 
Examples are gender, social class, blood type, country affiliation, observation time or rating via Likert scales."""

# What is quantitative variable in Dataframe?
"""quantitative variables are variables measured on a numeric scale. 
Height, weight, response time, subjective rating of pain, temperature, 
and score on an exam are all examples of quantitative variables."""

"""
-Create a DataFrame from a .csv file using the pd.read_csv function.
-Have a brief overview of the data (head method, columns and shape attributes).
-Select values in the DataFrame (loc and iloc methods).
-Carry out a quick statistical study of our data (describe and value_counts methods)"""

# Describing a numeric Series (quantitive).
# s = pd.Series([1, 2, 3])
# s.describe()
# count    3.0
# mean     2.0
# std      1.0
# min      1.0
# 25%      1.5
# 50%      2.0
# 75%      2.5
# max      3.0

# Describing a (categorical) Series.
"""To analyze categorical variables, it is best to start by using the value_counts method
# dataframe["exemple"].value_counts"""

# s = pd.Series(['a', 'a', 'b', 'c'])
# s.describe()
# count     4
# unique    3
# top       a
# freq      2
# dtype: object

"""-Select one or more columns of a DataFrame by entering their names between square brackets[] as for a dictionary."""
# print(transactions['cust_id'])
# Extraction two columns 'cust_id' et 'Qty'

"""for multi columns we use two square brackets [[]]"""
# cust_id_qty = transactions[["cust_id","Qty"]] (2--> [])
#
# cat_vars = transactions[['cust_id', 'tran_date', 'prod_subcat_code', 'prod_cat_code', 'Store_type']]
# cat_var_names = ['cust_id', 'tran_date', 'prod_subcat_code', 'prod_cat_code', 'Store_type']
# cat_vars = transactions[cat_var_names]

"""
-Select one or more rows of a DataFrame by entering their index using the loc and iloc methods.
-Select rows from a DataFrame that satisfy a specific condition using conditional indexing."""
# transactions_eshop = transactions.loc[transactions['Store_type'] == 'e-Shop']
"""
-Perform a quick statistical study of quantitative variables in a DataFrame using the **describe** method."""

"""Display the first 10 rows of the DataFrame."""
# df.head(10)

"""Display the last 10 rows of the DataFrame."""
# df.tail(10)

"""Displaying the columns of the DataFrame df"""
# print(df.columns)

"""Display of the dimensions of my_dataframe"""
# print(df.shape)

"""Display of the dimensions of rows"""
# votes.shape[0]

"""Display of the dimensions of columns"""
# votes.shape[1]

"""chose the index (index_col"""
# transactions = pd.read_csv("transactions.csv", sep=",", index_col= "transaction_id")

"""Selecting rows from a DataFrame: loc and iloc methods"""
"""**loc**"""
# print(num_vars.loc[80712190438])
#                  Rate    Tax  total_amt
# transaction_id
# 80712190438     -772.0  405.3    -4265.3
# 80712190438      772.0  405.3     4265.3

#           smartphone	chaussures	console
# prix	        1000	 100	    400
# enStock	    True	 False	    True

# df.loc['prix','chaussures']
# 100

# df.loc[['prix','enStock'],'chaussures']

# prix          100
# enStock       False

# transactions.loc[[80712190438, 29258453508], ['Tax', 'total_amt']]

# transaction_id	    Tax	        total_amt
# 80712190438	        405.300	    -4265.300
# 29258453508	        785.925	    8270.925

"""**iloc**"""

# The iloc method makes it possible to index a DataFrame exactly like a numpy array,
# that is to say by only filling in the numeric indexes of the rows and columns.
# This allows to use slicing without constraints:

# # Extraction of the first 4 rows and the first 3 columns of transactions
# transactions.iloc[0:4, 0:3] (4 rows, 3 columns) (transactions_id <-->index)

# transaction_id	    cust_id	    tran_date	    prod_subcat_code
# 80712190438	        270351	    28-02-2014	    1.0
# 29258453508	        270384	    27-02-2014	    5.0
# 51750724947	        273420	    24-02-2014	    6.0
# 93274880719	        271509	    24-02-2014	    11.0

"""we can use iloc to seperate the datframe into 2 datframes"""
# df1 = datasX.iloc[:, :72]
# df2 = datasX.iloc[:, 72:]
# -------------------------------------------------------------------------------------------
"""Conditional Indexing of a DataFrame"""
# As with Numpy arrays, we can use conditional indexing to extract rows from a Dataframe that satisfy a given condition.
# In the following illustration, we are selecting rows from DataFrame df where value in column (col 2) = 3.
# df[df['col 2'] == 3]

# df.loc[df['col 2'] == 3]

"""# if we want to change a value in dataframe using loc"""
# 	           date         Event
# 0	        28-02-2014	    Dance
# 1	        08-02-2014	    painting
# 2	        02-02-2014	    swimming

# df.loc[(df.Event == 'Dance'),'Event']='Hip-Hop'
# 	           date         Event
# 0	        28-02-2014	    Hip-Hop
# 1	        08-02-2014	    painting
# 2	        02-02-2014	    swimming
# --------------------------------------------------------------------------------------------

"""A column of a DataFrame can be iterated over as a list in a for loop"""

# total = 0
# for i in transactions_client_268819['total_amt']:
#     total +=i

# print(total)
# ------------------------------------------------------------------------
"""What is the average of the amount of transactions whose amount is positive"""
# transactions[transactions['total_amt']  > 0].describe()
# new = transactions[transactions['total_amt'] > 0]
# new.describe()
# new['total_amt'].mean()
# -------------------------------------------------------------------------------------------
"""This stage of preparing a dataset is always the first stage of a data project."""

# Regarding data cleaning, we have learned to:
"""
-Identify and remove duplicates from a DataFrame using the (**duplicated** and **drop_duplicates**) methods.
-Modify the elements of a DataFrame and their type using the (**replace**, **rename** and **astype**) methods.
-Apply a function to a DataFrame with the **(apply)** method and the **(lambda clause)**."""

# Regarding the management of missing values, we have learned to:
"""
-Detect them using the **(isna)** method followed by the any and sum methods.
-Replace them using the **(fillna)** method and statistical methods.
-Delete them using the **(dropna)** method.
"""
# -------------------------------------------------------------------------------------------
"""Duplicates management (**duplicated**and **drop_duplicates** methods)"""
"""Locate rows containing duplicates (duplicated() --> sum() -->)"""
# Age	Sexe	Taille
# Robert	56	M	174
# Mark	23	M	182
# Alina	32	F	169
# Mark	23	M	182

# df.duplicated()
# 0  False
# 1  False
# 2  False
# 3  True

"""To calculate the sum of booleans, we consider that True is 1 and False is 0"""
# print(df.duplicated().sum())
# output : 1 (ther's one duplicate)

"""The method of a DataFrame to remove duplicates is drop_duplicates.
Its header is as follows:"""
# drop_duplicates****(subset, keep, inplace)****

"""
-The **(subset)** parameter indicates the **(column)** or columns to consider to identify and remove duplicates. """
# -By default,**subset = None**: we consider **all the columns of the DataFrame**.

"""-The **(keep)** parameter indicates which entry should be kept:"""
# -'first': We keep the first occurrence.
# -'last': We keep the last occurrence.
# -'False': We do not keep any of the occurrences.
# - By default, keep = 'first'.

"""The **(inplace)** parameter (very common in the methods of the DataFrame class)
specifies whether the DataFrame is directly modified **(in this case inplace=True)** 
or whether the method returns a copy of the DataFrame **(inplace=False)**. """
# A method applied with the argument **(inplace = True=** is irreversible. **(By default, inplace = False)**.

# Age	Sexe	Taille
# Robert	56	M	174
# Mark	23	M	182
# Alina	32	F	169
# Mark	23	M	182

# df_first = df.drop_duplicates(keep = 'first')

# Age	Sexe	Taille
# Robert	56	M	174
# Mark	23	M	182
# Alina	32	F	169

# df_last = df.drop_duplicates(keep = 'last')

# Age	Sexe	Taille
# Robert	56	M	174
# Alina	32	F	169
# Mark	23	M	182

# df_false = df.drop_duplicates(keep = False)

# Age	Sexe	Taille
# Robert	56	M	174
# Alina	32	F	169


"""Example (subset, keep, inplace):"""
# import pandas as pd
# data = pd.read_csv("employees.csv")

# sorting by first name
# data.sort_values("First Name", inplace=True)

# dropping ALL duplicate values
# data.drop_duplicates(subset="First Name",keep=False, inplace=True)


"""Example (rename, replace, astype):"""
"""The **(replace)** method is used to replace one or more values of a column of a DataFrame."""
# 1) Replace a single value with a new value for an individual DataFrame column:
# df['column name'] = df['column name'].replace(['old value'],'new value')

# (2) Replace multiple values with a new value for an individual DataFrame column:
# df['column name'] = df['column name'].replace(['1st old value','2nd old value',...],'new value')

# (3) Replace multiple values with multiple new values for an individual DataFrame column:
# df['column name'] = df['column name'].replace(['1st old value','2nd old value',...],['1st new value','2nd new value',...])

# (4) Replace a single value with a new value for an entire DataFrame:
# df = df.replace(['old value'],'new value')

# (5) Replace values with a new values for an entire DataFrame:
# df = df.replace(("y","n"), (1,0))

# colors = {'first_set':  ['Green','Green','Green','Blue','Blue','Red','Red','Red'],
#           'second_set': ['Yellow','Yellow','Yellow','White','White','Blue','Blue','Blue']}

# df = pd.DataFrame(colors, columns= ['first_set','second_set'])
# df = df.replace(['Blue'],'Green')

# transactions= transactions.replace(to_replace =['e-Shop', 'TeleShop', 'MBR', 'Flagship store',  np.nan], value =[1, 2, 3, 4, 0] )

# #transactions["Store_type"]= transactions["Store_type"].replace(['e-Shop', 'TeleShop', 'MBR', 'Flagship store',  np.nan], [1, 2, 3, 4, 0] )


"""astype"""
# new_types = {'Store_type' : 'int','prod_subcat_code' : 'int'}
# transactions = transactions.astype(new_types)

# or
# transactions[['Store_type','prod_subcat_code']]= transactions[['Store_type','prod_subcat_code']].astype('int')

# or(verify)
# df[df.columns[1:]] = df.iloc[:, 1:].astype(int)

"""rename"""
# transactions = transactions.rename({'Store_type': 'store_type',
# 'Qty': 'qty',
# 'Rate': 'rate',
# 'Tax ': 'tax'}, axis = 1) #on chose the axis to change the columns names.

# or
# customer = customer.rename(columns = {'customer_Id':'cust_id'})
# -------------------------------------------------------------------------------------------
"""Operations on the values of a DataFrame **(apply)** method and **(lambda function)**"""

"""
-**func** is the function to apply to the column.
-**axis** is the dimension on which the operation must apply."""

"""apply"""
# using numpy:
# df_lines = df.apply(np.sum, axis = 0)


# or
# df = pd.DataFrame({'A': [1, 2], 'B': [10, 20]})

# def square(x):
#     return x * x

# df1 = df.apply(square)

# index     A   B
# 0         1  10
# 1         2  20

# index     A    B
# 0         1  100
# 1         4  400


"""split"""
# date = '28-02-2014'

# Split string on '-' character
# print(date.split('-'))
# ['28', '02', '2014']

# def get_day(string):
#     day = string.split("-")[0]
#     return day


# def get_month(string):
#     month = string.split("-")[1]
#     return month


# def get_year(string):
#     year = string.split("-")[2]
#     return year

"""we apply the (**apply method** on the(tran_date) column of dataframe """
# days = transactions["tran_date"].apply(get_day)
# months = transactions["tran_date"].apply(get_month)
# years = transactions["tran_date"].apply(get_year)

"""we create a new columns using the previous values"""
# transactions['day'] = days
# transactions['month'] = months
# transactions['year'] = years

# transactions.head()

"""lambda function"""
"""
Lambda functions allow you to define functions with a very short syntax.
"""

# The classic definition of a function is done with the def clause:

# def increment(x):
#    return x+1

"""It is also possible to define a function with the lambda clause:
increment=lambdax:x+1
"""

"""example"""
# def get_day(date):
#     day = date.split("-")[0]
#     return day

# days = transactions["tran_date"].apply(get_day)
# transactions['day'] = days

"""with lambda"""
# transactions['day'] = transactions['tran_date'].apply(lambda date: date.split('-')[0])

"""example"""
# transactions["prodt"] = transactions.apply(lambda row: row['total_amt'] / row['qty'], axis = 1)

"""example"""
# we groupby the Needs and we aggregate the score and ressource_id and separate each value with (;)
# ress_need_df_ca = ress_need_df_ca.groupby('besoinid').agg({'ressource_id': lambda x: ';'.join(x),
#                                          'score': lambda x: ';'.join(x)}

# ---------------------------------------------------------------------------------------------
"""Management of missing values **NaN** """

"""A missing value is **(NaN)**:"""
# - An unspecified value.
# - A value that does not exist

"""
-Detection of missing values **(isna)** and **(any)**methods.
-The replacement of these values **(fillna method)**.
-Deleting missing values **(dropna method)**."""

# **True** if the original array cell is a missing value (np.nan).
# **False** otherwise.

# df.isna()
#       Nom	    Pays	Age
# 0	    True	False	True
# 1	    False	False	False
# 2	    False	False	False

"""We detect the COLUMNS containing at least one missing value"""
# df.isna().any(axis = 0)

# Name    True
# Country False
# Age     True


"""We detect the LINES containing at least one missing value"""
# df.isna().any(axis = 1)

# 0  True
# 1  False
# 2  False


"""We use conditional indexing to display entries containing missing values"""
# df[df.isna().any(axis=1)]

#    Nom	Pays	    Age
# 0	 NaN	Australie	NaN

"""example"""  # any() important in conditional indexing
# transactions[transactions[['rate', 'tax' ,'total_amt']].isnull().any(axis = 1)].head(10)

# 	                    cust_id	 tran_date	 prod_subcat_code	prod_cat_code	qty	rate	tax	total_amt	store_type
# transaction_id
# 27576087298	        270419	 9-2-2014	 11.0	6	-2	    NaN	            NaN	            NaN	            MBR
# 6472413088	        272105	 31-01-2014	 11.0	6	2	    NaN	            NaN	            NaN	            e-Shop
# 13300797307	        272662	 12-1-2014	 6.0	5	4	    NaN	            NaN	            NaN	            TeleSh

"""example"""
# bateaux_non_reserves = bateaux_clients[bateaux_clients['nom_client'].isna()]

"""Count the number of missing values for each COLUMN"""
# df.isnull().sum(axis = 0) #The isnull and isna functions are strictly equivalent

# Name 1
# Country 0
# Age 1

"""Count the number of missing values for each LINE"""
# df.isnull().sum(axis=1)

# 0 2
# 1 0
# 2 0

# --------------------------------------------------------------------------------------------
"""Replacement of missing values (fillna method)"""

"""We replace all the NANs of the DataFrame by zeros"""
#  df.fillna(0)

"""We replace the NANs of each numerical column by the average on this column"""
# df.fillna(df.mean()) # df.mean() can be replaced by any statistical method.

# The mean: mean.
# The median: median.
# The minimum/maximum: min/max.

"""For categorical type columns, we will replace the missing values with:"""
# The mode, i.e. the most frequent modality: mode.
# An arbitrary constant or category: 0, -1


"""Example"""
# Replace missing values ​​in the **column**:
# transactions["prod_subcat_code"] = transactions["prod_subcat_code"].fillna(-1)


"""mode()"""
"""we can use mode( for categorical values_string)"""

"""Example"""
# store_type_mode = transactions['store_type'].mode()
# print(store_type_mode)
#  0  e-shop
# store_type_mode = transactions['store_type'].mode()[0]
# print(store_type_mode)
# e-shop

# transactions['store_type'] = transactions['store_type'].fillna(transactions['store_type'].mode()[0])
# transactions[['prod_subcat_code', 'store_type']].isna().sum()


# customer['Gender'] = customer['Gender'].fillna(customer['Gender'].mode()[0])
# customer['city_code'] = customer['city_code'].fillna(customer['city_code'].mode()[0])
"""Example"""
# df = pd.DataFrame(
#     {"Age": [43, 23, 43, 49, 71, 37], "Test_Score": [90, 87, 96, 96, 87, 79]})

# print(df)
#    Age  Test_Score
# 0   43          90
# 1   23          87
# 2   43          96
# 3   49          96
# 4   71          87
# 5   37          79

# print(df.mode())
#     Age  Test_Score
# 0  43.0          87
# 1   NaN          96

# print(df["Test_Score"].mode()[0])
# 87
# print(df["Test_Score"].mode()[1])
# 96

"""Removing missing values **(dropna method)**"""
"""
-The **axis** parameter specifies whether rows or columns should be deleted (0 for rows, 1 for columns).
-The **how** parameter is used to specify how rows (or columns) are deleted:
-**how = 'any':** We delete the row (or column) if it contains at least one missing value.
-**how = 'all':** We delete the line (or column) where all elements are missing.
"""
"""Delete all rows containing at least one missing value"""
# df = df.dropna(axis = 0, how = 'any')

"""Delete empty columns"""
# df = df.dropna(axis = 1, how = 'all')

"""We delete the rows with missing values in the 3 columns ['col2', 'col3', 'col4']"""
# df = df.dropna(axis = 0, how = 'all', subset = ['col2','col3','col4'])

"""example"""
# transactions = transactions.dropna(axis = 0, how = 'all', subset = ['rate', 'tax', 'total_amt'])
#  -------------------------------------------------------------------------------------------
"""Data processing"""

"""Data preprocessing can be summed up in the use of 4 essential operations:
-filter
-unite
-order
-group
"""

"""Filter a DataFrame with binary operators.

-The 'and' operator: **(&)**.
-The 'or' operator: **(|)**.
-The 'not' operator: **(-)**.
"""


#  in the past example we saw the conditional indexing:
# select rows that have the value of 3 in it.
# df[df['col 2'] == 3]

""" **and** (&) example"""
# 	    quartier	        annee	surface
# 0	    'Champs-Elysées'	1979	70
# 1	    'Europe'	        1850	110
# 2	    'Père-Lachaise'	    1935	55
# 3	    'Bercy'	            1991    30

# Filtering the DataFrame on the 2 previous conditions
"""The conditions must be filled in between parentheses ()&()"""
# print(df[(df["annee"] == 1979) & (df["surface"] > 60)])
#  index   quartier           annee  surface
#  0       Champs-Elysées     1979       70

# e_shop = transactions[(transactions['store_type']== 'e_shop')&(transactions['total_amt']>=5000)]


""" **or** (|) example"""  # (altgr + 6)
# 	    quartier	        annee	surface
# 0	    'Champs-Elysées'	1979	70
# 1	    'Europe'	        1850	110
# 2	    'Père-Lachaise'	    1935	55
# 3	    'Bercy'	            1991    30

# print(df[(df['année'] > 1900) | (df['quartier'] == 'Père-Lachaise')])

#     quartier           annee    surface
# 0   Champs-Elysées     1979       70
# 2   Père-Lachaise      1935       55
# 3   Bercy              1991       30


""" **no** (-) example"""
# 	    quartier	        annee	surface
# 0	    'Champs-Elysées'	1979	70
# 1	    'Europe'	        1850	110
# 2	    'Père-Lachaise'	    1935	55
# 3	    'Bercy'	            1991    30

"""DataFrame filtering on all citys **except**(without) Bercy"""
# print(df[-(df['quartier'] == 'Bercy')])
#      quartier         annee   surface
#  0   Champs-Elysées   1979       70
#  1   Europe           1850      110
#  2   Père-Lachaise    1935       55


"""Uniting DataFrames: **concat** function and **merge** method."""

"""Union of DataFrames with **concat**"""
# union = pd.concat([part_1,part_2], axis = 0)
# union


"""Union of DataFrames with **merge**
-Two DataFrames can be merged if they have a column in common.
-The **(right)** parameter is the DataFrame--> (df2) to merge with the one calling the method-->(df1).
-The **(on)** parameter is the name of the common columns that will serve as a reference for the merge.
-The **(how)** parameter can take 4 values ('inner', 'outer', 'left', 'right')."""


# df1.merge(right = df2, on = 'column', how = 'inner')
"""
-'inner': This is the default value of the how parameter. 
The inner join returns **rows** whose values in **common** columns are present in **both** DataFrames.
this type of join is often discouraged because it **can lead** to the **loss** of many entries.
On the other hand, the inner join does **not produce any NaN**
"""

# df1.merge(right = df2, on = 'column', how = 'outer')
"""
'outer': The outer join merges the entire two DataFrames. 
No line will be deleted. This method can generate a lot of NaNs."""

# df1.merge(right = df2, on = 'column', how = 'left')
"""
'left': The left join returns all rows from the left DataFrame, and pads them with rows from the
 second DataFrame that match according to the values of the common column."""

# df1.merge(right = df2, on = 'column', how = 'right')
"""
'right': The right join returns all the rows of the right DataFrame, and completes them with the rows
 of the left DataFrame which coincide according to the indices of the common column."""

# ------------------------------------------------------------------------------------------------
"""set and reset index"""
""" It is possible to redefine the index of a DataFrame using the **(set_index)** method."""
# df = df.set_index('Nom')


"""We can also define the index from a Numpy array or a Series, etc...:"""
# New index to use
# new_index = ['10000' + str(i) for i in range(6)]
# print(new_index)
# # ['100000', '100001', '100002', '100003', '100004', '100005']

# Using an array or a Series is equivalent
# index_array = np.array(new_index)
# index_series = pd.Series(new_index)

# df = df.set_index(index_array)
# df = df.set_index(index_series)

#           Nom	        Voiture
# 100000	Lila	    Twingo
# 100001	Tiago	    Clio
# 100002	Berenice	C4 Cactus
# 100003	Joseph	    Twingo
# 100004	Kader	    Swift
# 100005	Romy	    Scenic

"""To return to the default numeric indexing, we use the reset_index method of the DataFrame:"""
# df = df.reset_index()

#       index	    Nom	        Voiture
# 0	    100000	    Lila	    Twingo
# 1	    100001	    Tiago	    Clio
# 2	    100002	    Berenice	C4 Cactus
# 3	    100003	    Joseph	    Twingo
# 4	    100004	    Kader	    Swift
# 5	    100005	    Romy	    Scenic

"""in the df we have an index so we can use it to set the index to the df2"""
# index_df = df.index
# df2 = df2set_index(index_df)
# -------------------------------------------------------------------------------------------
"""Sort and order the values of a DataFrame: **(sort_values)** and **(sort_index)** methods."""

# Prenom	    Note	Points_bonus
# 'Amelie'	    A	    1
# 'Marin'	    F	    1
# 'Pierre'	    A	    2
# 'Zoe'	        C	    1

"""df_sorted= df.sort_values(by = 'Points_bonus', ascending = True)"""
# Prenom	    Note	Points_bonus
# 'Amelie'	    A	    1
# 'Marin'	    F	    1
# 'Zoe'	        C	    1
# 'Pierre'	    A	    2


# df_sorted= df.sort_values(by = ['Points_bonus','Note], ascending = True)
# Prenom	    Note	Points_bonus
# 'Amelie'	    A	    1
# 'Zoe'	        C	    1
# 'Marin'	    F	    1
# 'Pierre'	    A	    2

"""sort index"""
"""The sort_index method sorts a DataFrame according to its index."""

# # We define the 'Note' column as the index of df
"""df = df.set_index('Note')"""

# # We sort the DataFrame df according to its index
"""df = df.sort_index()"""

# Note	 Prenom	    Points_bonus
# A	    'Amelie'	1
# A	    'Pierre'	2
# C	    'Zoe'	    1
# F	    'Marin'	    1

# -------------------------------------------------------------------------------------------
"""Grouping elements of a DataFrame: **groupby**, **agg** and **crosstab** methods."""

"""
-The **(groupby)** method is used to **group the rows** of a DataFrame that **share a common** value on a column.
-The general structure of a groupby operation is as follows:
    -Separation of data (Split).
    -Application of a function (Apply).
    -Combination of results (Combine)."""
# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                               'Parrot', 'Parrot'],
#                    'Max Speed': [380., 370., 24., 26.]})
# print(df)
#    Animal  Max Speed
# 0  Falcon      380.0
# 1  Falcon      370.0
# 2  Parrot       24.0
# 3  Parrot       26.0

# df.groupby(['Animal']).mean()
#
# Animal      Max Speed
# Falcon      375.0
# Parrot       25.0

"""**agg** method"""
"""
-It is possible to specify for each column which function should be used in the Application 
step of a groupby operation.
-To do this, we use the **agg** method of the DataFrameGroupBy class by **giving** it 
a **dictionary** where each **key** is the name of a column and the **value** is the **function** to apply."""

# df = pd.DataFrame([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9],
#                    [np.nan, np.nan, np.nan]],
#                   columns=['A', 'B', 'C'])

# df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
#         A    B
# sum  12.0  NaN
# min   1.0  2.0
# max   NaN  8.0

"""example"""
# -Separate transactions by customer ID.(index)
# -For the total_amt column, calculate the minimum (min), maximum (max) and sum (sum).
# -For the store_type column, count the number of modalities taken (count).(type of store)
# -Combine the results into a DataFrame.

# transaction_id	cust_id	tran_date	prod_subcat_code	prod_cat_code	qty	    rate	tax	        total_amt	store_type
# 80712190438	    270351	28-02-2014	  1	                1	            -5	    -772	405.3	    -4265.3	    e-Shop
# 29258453508	    270384	27-02-2014	  5	                3	            -5	    -1497	785.925	    -8270.92	e-Shop
# 51750724947	    273420	24-02-2014	  6	                5	            -2	    -791	166.11	    -1748.11	TeleShop
# 93274880719       271509	24-02-2014	  11	            6	            -3	    -1363	429.345	    -4518.35	e-Shop
# 51750724947	    273420	23-02-2014	  6	                5	            -2	    -791	166.11	    -1748.11	TeleShop

# To find the number of modalities taken by the store_type column, we will use the following lambda function:

# import numpy as np
# n_modalities = lambda store_type: len(np. unique(store_type))

# functions_to_apply = {
#     'total_amt' : ['min', 'max', 'sum'],
#     'store_type' : n_modalities
# }

# transactions.groupby('cust_id').agg(functions_to_apply)


# 	                       total_amt               	 store_type

#                 min 	      max	    sum	             /
# cust_id
# 266783	    -5838.82	5838.82	    3113.89	         2
# 266784	    442	        4279.66	    5694.07	         3
# 266785	    -6828.9	    6911.77	    21613.8	         3
# 266788	    1312.74	    1927.12	    6092.9 7	     3
# 266794	    -135.915	4610.06	    27981.9	         4

"""example"""
# Using a groupby operation, determine for each customer from the quantity of items purchased in a transaction (qty column):
# The maximum quantity.
# The minimum quantity.
# The median amount.
# It will be necessary to filter the transactions for which the quantity is negative.
# For this, you can use conditional indexing (qty[qty > 0]) of the column in a lambda function.

# quantite_max = lambda qty :qty[qty > 0].max()
# quantite_min = lambda qty :qty[qty > 0].min()
# quantite_med = lambda qty :qty[qty > 0].mean()


# functions_to_apply = {
#     'qty' : [quantite_max, quantite_min, quantite_med]
# }

# qty_groupby = transactions.groupby('cust_id').agg(functions_to_apply)

# qty_groupby.head()

# For a better display, we can rename the columns produced by the groupby
# qty_groupby.columns.set_levels([quantite_max, quantite_min, quantite_med], level=1, inplace=True)

"""crosstab"""

"""Another way to group data is to use pandas' **crosstab** function 
which used to cross data from the columns of a DataFrame."""

"""
-The normalize argument of crosstab allows frequencies to be displayed as a percentage.
-The argument **normalize = 1** normalizes the array on axis 1, i.e. on each **column**
-by entering the argument **normalize = 0**, the table is normalized on each **line**"""

"""example"""
# colonne1 = transactions['tran_date'].apply(lambda x: x.split('-')[2]).astype("int")

# colonne2 = transactions['store_type']

# pd.crosstab(colonne1,
#             colonne2,
#             normalize = 1)

# store_type      Flagshipstore	MBR	        TeleShop	e-Shop

# tran_date
# 2011	        0.291942	    0.323173	0.283699	0.306947
# 2012	        0.331792	    0.322093	0.336767	0.322886
# 2013	        0.335975	    0.3115	    0.332512	0.320194
# 2014	        0.0402906	    0.0432339	0.0470219	0.0499731
# This DataFrame allows us to say that 33.5975% of transactions carried out in a 'Flagship store' took place in 2013.


# normalize = 0
# colonne2 = transactions['store_type']
# pd.crosstab(colonne1,
#             colonne2,
#             normalize = 0)

# store_type

# tran_date	Flagshipstore	MBR	        TeleShop	e-Shop
# 2011	        0.191121	0.21548	    0.182617	0.410781
# 2012	        0.20096	    0.198693	0.20056	    0.399787
# 2013	        0.205522	0.194074	0.2	        0.400404
# 2014	        0.173132	0.189215	0.198675	0.438978
# Line normalization allows us to deduce that transactions made in an 'e-Shop' account for 41.0781% of transactions in 2011.

"""example"""
# df = pd.read_csv("covid_tests.csv" , sep = ";",index_col = 'patient_id')

# tab1=df["test_result"]
# tab2=df["infected"]

# pd.crosstab(tab1,tab2,normalize = 1)

# infected	    0	            1
# test_result
# 0	            0.944444	0.040541
# 1	            0.055556	0.959459

# we can see that 0 with 0 (0,944)  the result is real with --> 94%
# we can see that 0 with 1 (0,0555)  the result is not real with --> 5.5%
# ----------------------------------------------------------------------------------------------------------
"""create data frame"""
# Create NumPy array
array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Instanciation d'un DataFrame
df = pd.DataFrame(
    data=array,  # The data to format
    index=["i_1", "i_2", "i_3"],  # The index of each entry
    columns=["A", "B", "C", "D"],  # columns names
)

print(df)
#    A   B   C   D
# i_1  1   2   3   4
# i_2  5   6   7   8
# i_3  9  10  11  12


# or
dictionary = {"A": [1, 5, 9], "B": [2, 6, 10], "C": [3, 7, 11], "D": [4, 8, 12]}

# Instanciation d'un DataFrame
new_df = pd.DataFrame(data=dictionary, index=["i_1", "i_2", "i_3"])

print(new_df)
#       A   B   C   D
# i_1   1   2   3   4
# i_2   5   6   7   8
# i_3   9  10  11  12
# ------------------------------------------------------------------------------------------------
dictionnaire = {
    "Produit": ["miel", "farine", "vin"],
    "Date d'expiration": ["10/08/2025", "25/09/2024", "15/10/2023"],
    "Quantité": [100, 55, 1800],
    "Prix à l'unité": [2, 3, 10],
}

df = pd.DataFrame(dictionnaire)

print(df)
# ----------------------------------------------------------------------------------------------
"""Creating a DataFrame from a data file"""

# df = pd.read_csv("transactions.csv", sep=",", header=0, index_col="transaction_id")
# print(df)
# --------------------------------------------------------------------------------------------
""" Normalization """

df = pd.DataFrame(
    [
        [180000, 110, 18.9, 1400],
        [360000, 905, 23.4, 1800],
        [230000, 230, 14.0, 1300],
        [60000, 450, 13.5, 1500],
    ],
    columns=["Col A", "Col B", "Col C", "Col D"],
)

# view data
print(df)


# copy the data
df_max_scaled = df.copy()

# apply normalization techniques
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

# view normalized data
print(df_max_scaled)


# copy the data
df_min_max_scaled = df.copy()

# apply normalization techniques
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (
        df_min_max_scaled[column] - df_min_max_scaled[column].min()
    ) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

# view normalized data
print(df_min_max_scaled)


# copy the data
df_z_scaled = df.copy()

# apply normalization techniques
for column in df_z_scaled.columns:
    df_z_scaled[column] = (
        df_z_scaled[column] - df_z_scaled[column].mean()
    ) / df_z_scaled[column].std()

# view normalized data
print(df_z_scaled)

cols = ["Col A", "Col B", "Col C", "Col D"]
# or
df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

df.head()

# or
df = (df - df.min()) / (df.max() - df.min())

df.head()
# ------------------------------------------------------------------------------------------------
# def normalize(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result
