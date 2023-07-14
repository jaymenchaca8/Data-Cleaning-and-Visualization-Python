from distutils import core
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.set_option('display.max_columns', None)

df = pd.read_csv('D:\Datasets\moviesKaggle.csv')

####Data Cleaning####

#Check for missing data
for col in df.columns:
    pct_missing = np.mean(df[col].isnull()) * 100
    print('{} - {}%'.format(col,pct_missing))

"""
name - 0.0%
rating - 1.004173187271779%
genre - 0.0%
year - 0.0%
released - 0.02608242044861763%
score - 0.03912363067292645%
votes - 0.03912363067292645%
director - 0.0%
writer - 0.03912363067292645%
star - 0.013041210224308816%
country - 0.03912363067292645%
budget - 28.31246739697444%
gross - 2.464788732394366%
company - 0.2217005738132499%
runtime - 0.05216484089723526%
"""

#remove rows with missing data
df.dropna(axis = 0, inplace = True)

#print(df.dtypes)

"""
name         object
rating       object
genre        object
year          int64
released     object
score       float64
votes       float64
director     object
writer       object
star         object
country      object
budget      float64
gross       float64
company      object
runtime     float64
"""

#change some fields to a datatype that makes more sense
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

#correct release year
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)

#sort and display table
print(df.sort_values(by=['gross'], inplace = False, ascending = False))

#removing dupicates if needed
#df.drop_duplicates()

####Visualization and Correlations####

#gross vs budget w/ linear regression fit
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Gross vs Budget Earnings')
plt.ylabel('Gross Earnings')
plt.xlabel('Budget for Film')
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color": "red"}, line_kws={"color":"blue"})
plt.show()

#numeric correlations as a heatmap
correlation_matrix = df.corr(method = 'pearson',numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

#most relevant correlations in list-form
correlation_list = correlation_matrix.unstack()
high_corr = correlation_list[((correlation_list) > 0.5) & ((correlation_list) != 1)].drop_duplicates() #find high correlations that are not against the same variable
print(high_corr.sort_values())

'''
votes   gross          0.614751
budget  gross          0.740247
year    yearcorrect    0.998726
dtype: float64
'''



