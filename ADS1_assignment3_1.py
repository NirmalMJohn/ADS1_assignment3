import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import errors as err
from sklearn.linear_model import LinearRegression
def reader(file):
    '''
    It is defined to pass on the file name of the data frame, then later read
    the data frame and transpose it. Later the Original data frame & transposed
    data frame is returned
    Parameters
    ----------
    file :It holds the name of World Bank Data set which needs to be
    transposed further operations are being carried out.

    Returns
    -------
    df : Original Data Frame of the World Bank Data
    dft : Transposed Data frame
    '''
    df = pd.read_csv(file, skiprows=4)
    dft = df.transpose()
    header = dft.iloc[0]
    dft = dft.iloc[1:]
    dft.columns = header
    print(dft)
    return df, dft


def desired_countries(df):
    '''
    This function is used to filter out desired countries mentioned in the list
    by it country codes

    Parameters
    ----------
    df : Original Dataframe

    Returns
    -------
    df : Data after filteration of desired countries is returned

    '''
    values = []
    values = ['IND', 'CHN', 'USA', 'UKR', 'ETH', 'PRK', 'IDN', 'EGY', 'ARE',
              'RUS', 'IRQ', 'BRA', 'COG', 'KEN', 'FIN', 'EST', 'CHE', 'ZMB']
    df =  df[df['Country Code'].isin(values) ==
             True].reset_index().drop(['index'], axis=1)
    return df


def desired_indicators(df):
    '''
    This function is used to filter out our desired indicators by Indicator
    code given in the list

    Parameters
    ----------
    df : Data frame with desired countries is passed

    Returns
    -------
    df : Data set with desired countries & Indicators gets returned.

    '''
    values = []
    values = ['SP.URB.TOTL.IN.ZS', 'SP.URB.TOTL', 'SP.POP.TOTL',
              'EG.FEC.RNEW.ZS', 'EG.ELC.NGAS.ZS',
              'SI.POV.DDAY', 'EN.ATM.CO2E.KD.GD']
    df = df[df['Indicator Code'].isin(values) ==
            True].reset_index().drop(['index'], axis=1)
    return df


def data_clean(df):
    '''
    Data Cleaning is carried out in this function, All the null values are
    replaced by the row average or 0.1 in rows where there are no values.

    Parameters
    ----------
    df : Dataframe returned from desired_indicators() is passed as an arguement
    here

    Returns
    -------
    df : Cleaned data is returned.

    '''
    df1 = df[df.columns[df.columns.isin(['Country Name', 'Country Code',
                                         'Indicator Name'])]]

    df = df.drop(['Country Name', 'Country Code', 'Indicator Name'], axis=1)
    df.to_csv('ncl.csv')
    m = df.mean(axis=1)
    for i, col in enumerate(df):
        # using i allows for duplicate columns
        # inplace *may* not always work here, so IMO the next line is preferred
        # df.iloc[:, i].fillna(m, inplace=True)
        df.iloc[:, i] = df.iloc[:, i].fillna(m)
    # Some rows with complete null values remains the same,
    # For which we insert 0.1.
    df = df.replace(np.nan, 0.1)
    print(df.isna().sum())
    df.to_csv('cl.csv')
    df = pd.concat([df1, df], axis=1)
    return df


def data_preprocess2(df, value):
    '''
    Cluster preprocessing for Clustering the Countries over desired indicators
    from 1990-2021 is carried out in this function. This function is called
    in for loop with range of values .i.e. Country Code. For each country
    Average of each attribute in said period is year is returned.
    Parameters
    ----------
    df :  Cleaned Data
    value : It has Country codes

    Returns
    -------
    Final: After each loop Average of each indicator for respective country
    code is returned.

    '''
    df = df.groupby(['Country Code']).get_group(value)
    df = df.reset_index().drop(['index'], axis=1)
    df1 = df[df.columns[df.columns.isin(['Country Name', 'Country Code',
                                         'Indicator Name', 'Indicator Code'])]]
    df = df.drop(['Country Name', 'Country Code', 'Indicator Name',
                  'Indicator Code'], axis=1)
    df.drop(df.iloc[:, 0:31], inplace=True, axis=1)
    result = pd.DataFrame()
    result['Average'] = df.mean(axis=1)
    result = pd.concat([df1, result], axis=1)
    result = result.drop(['Country Name', 'Country Code',
                          'Indicator Code'], axis=1)

    result1 = result.T
    header = result1.iloc[0]
    result1 = result1.iloc[1:]
    result1.columns = header
    result1 = result1.reset_index().drop(['index'], axis=1)
    result1.to_csv('res1.csv')
    print(result1)
    r = result1
    final = pd.DataFrame()
    final = final.append(r)
    return final


def data_preprocess1(df, value):
    '''
    Cluster preprocessing for Clustering the Countries over desired indicators
    from 1960-1990 is carried out in this function. This function is called
    in for loop with range of values .i.e. Country Code. For each country
    Average of each attribute in said period is year is returned.
    Parameters
    ----------
    df : Cleaned Data
    value : it has country codes

    Returns
    -------
    Final: After each loop Average of each indicator for respective country
    code is returned.

    '''
    df = df.groupby(['Country Code']).get_group(value)
    df = df.reset_index().drop(['index'], axis=1)
    df1 = df[df.columns[df.columns.isin(['Country Name', 'Country Code',
                                         'Indicator Name', 'Indicator Code'])]]
    df = df.drop(['Country Name', 'Country Code', 'Indicator Name',
                  'Indicator Code'], axis=1)
    columns = df.columns
    col_name = '1990'
    column_index = columns.get_loc(col_name)
    print("Index of the column ", col_name, " is: ", column_index)
    df.drop(df.iloc[:, 31:], inplace=True, axis=1)
    result = pd.DataFrame()
    result['Average'] = df.mean(axis=1)
    result = pd.concat([df1, result], axis=1)
    result = result.drop(['Country Name', 'Country Code',
                          'Indicator Code'], axis=1)

    result2 = result.T
    header = result2.iloc[0]
    result2 = result2.iloc[1:]
    result2.columns = header
    result2 = result2.reset_index().drop(['index'], axis=1)
    result2.to_csv('res1.csv')
    print(result2)
    r = result2
    final = pd.DataFrame()
    final = final.append(r)
    return final


def norm_func(i):
    '''
    This function is used for normalizing the data before PCA Transformation

    Parameters
    ----------
    i : Data point in an array

    Returns
    -------
    x : After Normalization data is returned

    '''
    x = (i-i.min()) / (i.max() - i.min())
    return (x)





def exp_growth(t, scale, growth):
    
    f = scale * np.exp(growth * (t-1990))
    return f


# Passing the World Bank Data name as the arguements to reader function
data, datat = reader('API_19_DS2_en_csv_v2_4773766.csv')
# For fetching the data of our desired countries based Income category
new_data = desired_countries(data)
#  Again fetching the data on the basis of indicators relevant to subject
new_data1 = desired_indicators(new_data)
print(new_data1)
new_data1.to_csv('nd1.csv')
# Checking for Null values in data set
print(new_data1.isna().sum())  # We are dealing with quite a number of null
# values in the data set.
new_data1 = data_clean(new_data1)
print(new_data1)
cluster1 = pd.DataFrame()
first_half = pd.DataFrame()


values = []
values = ['IND', 'CHN', 'USA', 'UKR', 'ETH', 'PRK', 'IDN', 'EGY', 'ARE',
          'RUS', 'IRQ', 'BRA', 'COG', 'KEN', 'FIN', 'EST', 'CHE', 'ZMB']

for i in range(len(values)):
    first_half = data_preprocess1(new_data1, values[i])
    cluster1 = cluster1.append(first_half, ignore_index=True)

print(cluster1)
cluster1.to_csv('c1.csv')

norm1 = norm_func(cluster1.iloc[:, :])
print(norm1)

# Using Elbow Method & Silhouette Score for finding Number of cluster
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf1 = KMeans(n_clusters=i)
    clf1.fit(norm1)
    WCSS.append(clf1.inertia_)  # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()

for c in range(2, 10):
    model1 = KMeans(n_clusters=c)
    model1.fit(norm1)
    pred1 = model1.predict(norm1)
    score = silhouette_score(norm1, pred1)
    print('Silhouette Score for c = {}: {:<.3f}'.format(c, score))

# Considering both We find number of  Clusters=3 are optimum
clf1 = KMeans(n_clusters=3)
clf1.fit(norm1)
labels1 = clf1.predict(norm1)
l1 = pd.Series(labels1)
cluster1['Cluster'] = l1
print(cluster1)

# Inserting Country Name Columns
new_col = ['India', 'China', 'USA', 'Ukraine', 'Ethiopia', 'North Korea',
           'Indonesia', 'Egypt', 'UAE', 'Russia', 'Iraq', 'Brazil', 'Congo',
           'Kenya', 'Finland', 'Estonia', 'Switzerland', 'Zambia']

# Inserting the column at the middle of the DataFrame
cluster1.insert(loc=0, column='Country Name', value=new_col)
# show the dataframe
print(cluster1)

cluster2 = pd.DataFrame()
second_half = pd.DataFrame()

for i in range(len(values)):
    second_half = data_preprocess2(new_data1, values[i])
    cluster2 = cluster2.append(second_half, ignore_index=True)

norm2 = norm_func(cluster2.iloc[:, :])
print(norm2)


# Considering both We find number of  Clusters=3 are optimum
labels2 = clf1.predict(norm2)
l2 = pd.Series(labels2)
cluster2['Cluster'] = l2
print(cluster2)
# Inserting the column at the middle of the DataFrame
cluster2.insert(loc=0, column='Country Name', value=new_col)
# show the dataframe
print(cluster2)
c2 = [2]
c1 = [1]
c0 = [0]
x2 = cluster1[cluster1['Cluster'].isin(c2) == True].reset_index().drop(
    ['index'], axis=1)
x1 = cluster1[cluster1['Cluster'].isin(c1) == True].reset_index().drop(
    ['index'], axis=1)
x0 = cluster1[cluster1['Cluster'].isin(c0) == True].reset_index().drop(
    ['index'], axis=1)
plt.figure(dpi=144)
plt.scatter(x0['Urban population (% of total population)'],
            x0['CO2 emissions (kg per 2015 US$ of GDP)'],
            marker='^', label='0')
plt.scatter(x1['Urban population (% of total population)'],
            x1['CO2 emissions (kg per 2015 US$ of GDP)'], marker='o', label='1')
plt.scatter(x2['Urban population (% of total population)'],
            x2['CO2 emissions (kg per 2015 US$ of GDP)'], marker='*', label= '2')
plt.title('Cluster formation of countries during the year 1960-1990')
plt.ylabel('CO2 emissions (kg per 2015 US$ of GDP)')
plt.xlabel('Urban population (% of total population)')
plt.text(21.5073, 1.11836, 'India')
plt.text(19.4305, 1.30487, 'China')
plt.text(73.315, 1.377005, 'USA')
plt.text(57.8526, 3.38655, 'Ukraine')
plt.text(9.50655, 0.197001, 'Ethiopia')
plt.text(53.0189, 0.1, 'North Korea')
plt.text(20.5202, 0.628018, 'Indonesia')
plt.text(42.1468, 0.704202, 'Egypt')
plt.text(79.0162, 0.496258, 'UAE')
plt.text(65.3937, 1.65408, 'Russia')
plt.text(59.8074,1.21909,'Iraq')
plt.text(60.515, 0.24286, 'Brazil')
plt.text(43.4377, 0.533122, 'Congo')
plt.text(12.561, 0.208624, 'Kenya')
plt.text(67.6932, 0.284016, 'Finland')
plt.text(66.4561, 0.938008, 'Estonia')
plt.text(73.5735, 0.0769481, 'Switzerland')
plt.text(32.6591, 0.259566, 'Zambia')

plt.legend()
plt.show()

x2 = cluster2[cluster2['Cluster'].isin(c2) == True].reset_index().drop(
    ['index'], axis=1)
x1 = cluster2[cluster2['Cluster'].isin(c1) == True].reset_index().drop(
    ['index'], axis=1)
x0 = cluster2[cluster2['Cluster'].isin(c0) == True].reset_index().drop(
    ['index'], axis=1)
plt.figure(dpi=144)
plt.scatter(x0['Urban population (% of total population)'],
            x0['CO2 emissions (kg per 2015 US$ of GDP)'],
            marker='^', label='0')
plt.scatter(x1['Urban population (% of total population)'],
            x1['CO2 emissions (kg per 2015 US$ of GDP)'], marker='o', label='1')
plt.scatter(x2['Urban population (% of total population)'],
            x2['CO2 emissions (kg per 2015 US$ of GDP)'], marker='*', label='2')
plt.title('Cluster formation of countries during the year 1991-2021')
plt.ylabel('CO2 emissions (kg per 2015 US$ of GDP)')
plt.xlabel('Urban population (% of total population)')

plt.text(29.9016, 1.11216, 'India')
plt.text(44.1879, 1.25083, 'China')
plt.text(79.8394, 1.36922, 'USA')
plt.text(68.048, 3.32804, 'Ukraine')
plt.text(16.6743, 0.19849, 'Ethiopia')
plt.text(60.1858, 0.1, 'North Korea')
plt.text(47.9445, 0.633247, 'Indonesia')
plt.text(42.9002, 0.800353, 'Egypt')
plt.text(82.5578, 0.499354, 'UAE')
plt.text(73.7153, 1.64017, 'Russia')
plt.text(69.2768, 1.23132, 'Iraq')
plt.text(75.436, 0.24667, 'Brazil')
plt.text(61.4458, 0.530556, 'Congo')
plt.text(22.2864, 0.204081, 'Kenya')
plt.text(83.1505, 0.27963, 'Finland')
plt.text(69.1401, 0.93008, 'Estonia')
plt.text(73.6325, 0.0759386, 'Switzerland')
plt.text(38.9558, 0.250168, 'Zambia')

plt.legend()
plt.show()
lpdata = new_data1
# Plot the line Graph for Urbanization from Desired Population
Urban = lpdata.groupby(['Indicator Code']).get_group('SP.URB.TOTL.IN.ZS')
Urban = Urban.reset_index().drop(['index'], axis=1)

Urban = Urban.drop(columns=['Country Code', 'Indicator Name',
                            'Indicator Code'], axis=1)


Urban_T = Urban.set_index('Country Name').T.reset_index()
Urban_T = Urban_T.rename_axis(None, axis=1)
Urban_T.rename(columns={'index': 'Year'}, inplace=True)

plt.figure(figsize=(10, 8))
# Get current axis
plt.style.use('ggplot')
ax = plt.gca()
Urban_T.plot(kind='line', x='Year', y='China', ax=ax)
Urban_T.plot(kind='line', x='Year', y='Russian Federation', ax=ax)
Urban_T.plot(kind='line', x='Year', y='Indonesia', ax=ax)
Urban_T.plot(kind='line', x='Year', y='Ukraine', ax=ax)
Urban_T.plot(kind='line', x='Year', y='United States', ax=ax)
Urban_T.plot(kind='line', x='Year', y='India', ax=ax)
Urban_T.plot(kind='line', x='Year', y='Egypt, Arab Rep.', ax=ax)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.ylabel('Urbanization')
plt.title('Urbanization of Selected Countries 1960-2021')
plt.savefig('line.png', dpi=400)
# show the plot
plt.show()


# Plot Bar Graph of Electricity produced from Nuclear Energy
countries=['RUS','USA','UKR']
bpdata = new_data
nuclear = bpdata.groupby(['Indicator Code']).get_group('EG.ELC.NUCL.ZS')
nuclear = nuclear[nuclear['Country Code'].isin(countries) ==
                  True].reset_index().drop(['index'], axis=1)
nuclear = nuclear.dropna(axis=1)
nuclear = nuclear.drop(columns=['Country Code', 'Indicator Name',
                                'Indicator Code'], axis=1)
print(nuclear)
nuclear_t = nuclear.T

clear_t = nuclear_t.rename_axis(None, axis=1)
nuclear_t = nuclear_t.reset_index()
header = nuclear_t.iloc[0]
nuclear_t = nuclear_t.iloc[1:]
nuclear_t.columns = header
nuclear_t.rename(columns={'Country Name': 'Year'}, inplace=True)
print(nuclear_t)

nu_new = pd.DataFrame()
years = []
years = ['1990', '1995', '2000', '2005', '2010', '2014']
nu_new = nuclear_t.loc[nuclear_t['Year'].isin(years)].reset_index().drop(
    ['index'], axis=1)

print(nu_new)

plt.figure(figsize=(10, 8))

x_axis = np.arange(len(nu_new['Year']))

plt.bar(x_axis - 0.15, nu_new['Russian Federation'], width=0.1, label='Russia')
plt.bar(x_axis + 0.0, nu_new['Ukraine'], width=0.1, label='Ukraine')
plt.bar(x_axis + 0.15, nu_new['United States'],
        width=0.1, label='United States')

plt.title('Nuclear energy Harnessing of US,Ukraine & Russia')
plt.xlabel('Years')
plt.ylabel('Nuclear Energy harnessing in % of total')
plt.legend()
plt.xticks(x_axis, nu_new['Year'])
plt.show()

# For Curve fitting & error range

ukraine= nuclear_t.drop(columns=['Russian Federation','United States'],axis=1)
print(ukraine)
ukraine=ukraine.to_numpy()
x = ukraine[:,0].astype(int)  # Storing Independent feature Rainfall to x
y =ukraine[:,1] # Stroing dependent feature Productivity to y

print(x)
# Plotting the data as two dimensional scatter plot

plt.figure(dpi=144, figsize=(15,8))
plt.scatter(x, y)
plt.xlabel('Year')
plt.ylabel('Nuclear Energy Harnessing(% of Total)')
plt.title('Nuclear Energy harnessing for Ukraine 1990-2014')

popt, covar = opt.curve_fit(exp_growth, x,y, p0=[20, 0.016])

print("Fit parameter", popt)
z = exp_growth(x, *popt)
plt.figure(figsize=(15,8))
plt.plot(x, y, label="data")
plt.plot(x, z, label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Final fit exponential growth")
plt.show()
print()

#extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(x, exp_growth, popt, sigma)
plt.figure(figsize=(18,5))
plt.title("Exponential Fit")
plt.plot(x, y, label="data")
plt.plot(x, z, label="fit")
plt.fill_between(x, low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("Nuclear Energy Harnessing")
plt.show()


#Predicting the Nuclear Energy Harnessing by 2023

print("Forcasted population")
low, up = err.err_ranges(2023, exp_growth, popt, sigma)
print('The Predicted value for Nuclear Harnessing by Ukraine in 2023\
      will be between',low,'&',up)

#Plotting the prediction
x_new=x.astype(int)
x_new = np.append(x_new,[2016,2017,2018,2019,2020,2021,2022,2023])
z_new = exp_growth(x_new, *popt)
low, up = err.err_ranges(x_new, exp_growth, popt, sigma)
plt.figure(figsize=(20,7))
plt.title("Predicted Values for Ukraine's Nuclear Energy Harnessing in 2023")
plt.plot(x, y, label="data")
plt.plot(x_new, z_new, label="fit")
plt.fill_between(x_new, low, up, alpha=0.7)
plt.plot(2023,51.421, marker='o')
plt.plot(2023,68.607, marker='o')
plt.text(2023,51.421, 'lower limit = 51.42%')
plt.text(2023,68.607, 'Upper Limit = 68.60%')
plt.legend()
plt.xlabel("year")
plt.ylabel("Nuclear Energy Harnessing % of Total")
plt.show()


