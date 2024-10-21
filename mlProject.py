from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np 
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

##This function takes in the Data Frame and returns 2 Data Frames (test and train) using random permutation
##The problem with this method is that it doesn't allow for regularity, even if the random seed is constant, as 
##when data is added to the original data frame, you will get vastly different test and train splits, leading to 
##data snooping bias.
def shuffel_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

##Tests to see if the hash of the ID is within the range to be in the test set range
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

##Returns the two Data Frames as test and train Data Frames using the ID hash method
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
    
housing = load_housing_data()

##Preliminary information about inital data
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())

train_set, test_set = shuffel_and_split_data(housing, 0.2)


#housing.hist(bins=50, figsize=(12,8))
#plt.show()

#housing_with_id = housing.reset_index()
#train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
#print(test_set.head())
#print(train_set.head())

#print(train_set.info())
#print(test_set.info())

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

#print(test_set.head())
#print(train_set.head())

#print(train_set.info())
#print(test_set.info())

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1,2,3,4,5])

#print(housing.head())
#print(housing.info())

#print(housing["income_cat"].value_counts())

#housing["income_cat"].value_counts().plot.bar(rot=0, grid=True)
#plt.xlabel("Income Category")
#plt.ylabel("Number of Districts")
#plt.show()

#splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
#for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#    strat_train_set_n = housing.iloc[train_index]
#    strat_test_set_n = housing.iloc[test_index]
#    strat_splits.append([strat_train_set_n, strat_test_set_n])
#
#strat_train_set_n, strat_test_set_n = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#print(len(strat_test_set))
#print(strat_test_set["income_cat"].value_counts())

#housing = train_set.copy()
#
#housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
#plt.show()
#
#housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
#             s=housing["population"] / 100, label="population",
#             c="median_house_value", cmap="jet", colorbar=True,
#             legend=True, sharex=False, figsize=(10, 7))
#plt.show()

corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(9,6))
#plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
#plt.show()

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["room_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
#imputer = SimpleImputer(strategy="mean")
#imputer = SimpleImputer(strategy="most_frequent")

housing_num = housing.select_dtypes(include=[np.number])
#imputer.fit(housing_num)

#print(imputer.statistics_)
#print(housing_num.median().values)

#x = imputer.transform(housing_num)
X = imputer.fit_transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#Encode Text Data to Numbers
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]
#print(housing_cat_encoded[:8])

#print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

#min max scaling for housing_num
min_max_scalar = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scalar.fit_transform(housing_num)

# standardization of housing_num
std_scalar = StandardScaler()
housing_num_std_scaled = std_scalar.fit_transform(housing_num)

target_scalar = StandardScaler()
scaled_labels = target_scalar.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scalar.inverse_transform(scaled_predictions)

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)