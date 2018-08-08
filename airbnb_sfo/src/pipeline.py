import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error, r2_score
from collections import Counter

import statsmodels.api as sm


def price_xform(dframe,colname):
    dframe[colname]=dframe[colname].str.replace('$','')
    dframe[colname]=dframe[colname].str.replace(',','')
    dframe[colname]=pd.to_numeric(dframe[colname])
#     return(dframe)

def cat_rename(dframe,colname,src,tgt):
    dframe[colname]=dframe[colname].str.replace(src,tgt)
#     dframe[colname]=dframe[colname].str.replace(',','')
#     dframe[colname]=pd.to_numeric(dframe[colname])
#     return(dframe)

def create_dummies(dframe,colname):
#     dfnew=dframe.copy()
    dframe=pd.get_dummies(data=dframe, columns=[colname])# create dummies and drop the parent column
    dframe.drop(columns=[dframe.columns[-1]],inplace=True) #dropped the last column from the add dummies
    return dframe

def impute_nullrows(dframe,colname):
#Impute Null Rows based on the specified column
    print(dframe[colname].isnull().sum())
    dftmp=dframe[~dframe[colname].isnull()]   # remove nulls from one column
    print(dftmp[colname].isnull().sum())
    return dftmp

def init():
    # pd.set_option('display.max_columns', 96)
    pd.set_option('display.max_rows', 96)

def read_data():
    print('\nREADING INPUT DATASET .................')
    cal = pd.read_csv('calendar.csv.gz')
    listd = pd.read_csv('listings.csv.gz')
    lists = pd.read_csv('listings.csv')
    revs = pd.read_csv('reviews.csv.gz')
    nhood = pd.read_csv('neighbourhoods.csv')
    purelst = pd.read_csv('listings.csv.gz')
    # print(lists.head())

    print(cal.date.min(),cal.date.max())
    print(revs.date.min(),revs.date.max())

    return(listd)

def process_data(listd):
    print('\nPROCESSING INPUT DATASET .................')
    print(df.head())

    #!!! APPLY the same for the test data also

    #CLEANING UP INPUT DATASET
    listd_drop_cols = ['scrape_id','last_scraped','experiences_offered','thumbnail_url','medium_url','xl_picture_url',\
                   'host_name','host_location','neighbourhood_group_cleansed','square_feet',\
                   'maximum_nights','is_business_travel_ready']
    # listd_numc_cols  # Numeric Columns
    # listd_catz_cols  # Categorical Columns
    # print('Shape of Input Dataset',len(listd.columns), len(listd_drop_cols))
    listd.drop(listd_drop_cols,axis=1,inplace=True )
    # print('Shape of Output Dataset'len(listd.columns), len(listd_drop_cols))

    # Convert to Booleans
    listd['host_is_superhost'] = listd.apply(lambda x:  x.host_is_superhost=='t', axis= 1)
    listd['host_has_profile_pic'] = listd.apply(lambda x:  x.host_has_profile_pic=='t', axis= 1)
    listd['host_identity_verified'] = listd.apply(lambda x:  x.host_identity_verified=='t', axis= 1)
    listd['instant_bookable'] = listd.apply(lambda x:  x.instant_bookable=='t', axis= 1)

    #Transform all the price columns to remove "$" and ","
    price_xform(listd,'price')
    price_xform(listd,'weekly_price')
    price_xform(listd,'monthly_price')
    price_xform(listd,'security_deposit')
    price_xform(listd,'cleaning_fee')
    price_xform(listd,'extra_people')

    #Transform the room_type . to create a meaningful name and create dummies
    cat_rename(listd,'room_type','Entire home/apt','full')
    cat_rename(listd,'room_type','Private room','pvt')
    cat_rename(listd,'room_type','Shared room','shared')
    listd=create_dummies(listd,'room_type')

    #Transform the bed_type . to create a meaningful name and create dummies
    listd['bed_type'] = listd.apply(lambda x: x.bed_type=='Real Bed', axis=1)

    #Put a count on amenities,   IF it does not work, pick the most important feature
    listd['amentcnt'] =   listd.apply(lambda x: len(x.amenities.split(",")), axis=1)

    #Impute the to remove the rows with no zipcode.  !!! APPLY the same for the test data also
    listd=listd[~listd['zipcode'].isnull()]   # remove nulls from one column

    #Impute the null values with meaningful data
    listd['cleaning_fee']=listd['cleaning_fee'].fillna(0)
    listd['review_scores_cleanliness'] = listd['review_scores_cleanliness'].fillna(listd.review_scores_cleanliness.mean())
    listd['review_scores_location'] = listd['review_scores_location'].fillna(listd.review_scores_location.mean())
    listd['review_scores_value'] = listd['review_scores_value'].fillna(listd.review_scores_value.mean())

    listd.loc[listd['minimum_nights']<=7,'min_night_stay'] = 'short'
    listd.loc[(listd['minimum_nights']>7) & (listd['minimum_nights']<=32),'min_night_stay'] = 'mid'
    listd.loc[listd['minimum_nights']>32,'min_night_stay'] = 'long'
    listd=create_dummies(listd,'min_night_stay')

    #Condensing the property_type to sub categories
    listd.loc[listd['property_type']=='Apartment', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Condominium', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Guest suite', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Townhouse', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Guesthouse', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Tiny house', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Timeshare', 'ppt_condensed'] = 'apt'
    listd.loc[listd['property_type']=='Serviced apartment', 'ppt_condensed'] = 'aptspl'
    listd.loc[listd['property_type']=='Bed and breakfast', 'ppt_condensed'] = 'aptspl'
    listd.loc[listd['property_type']=='Treehouse', 'ppt_condensed'] = 'aptspl'
    listd.loc[listd['property_type']=='Cabin', 'ppt_condensed'] = 'aptspl'
    listd.loc[listd['property_type']=='Bus', 'ppt_condensed'] = 'auto'
    listd.loc[listd['property_type']=='Boat', 'ppt_condensed'] = 'auto'
    listd.loc[listd['property_type']=='Camper/RV', 'ppt_condensed'] = 'auto'
    listd.loc[listd['property_type']=='Hostel', 'ppt_condensed'] = 'hostel'
    listd.loc[listd['property_type']=='Boutique hotel', 'ppt_condensed'] = 'hotel'
    listd.loc[listd['property_type']=='Hotel', 'ppt_condensed'] = 'hotel'
    listd.loc[listd['property_type']=='Resort', 'ppt_condensed'] = 'hotel'
    listd.loc[listd['property_type']=='Aparthotel', 'ppt_condensed'] = 'hotel'
    listd.loc[listd['property_type']=='House', 'ppt_condensed'] = 'house'
    listd.loc[listd['property_type']=='Bungalow', 'ppt_condensed'] = 'house'
    listd.loc[listd['property_type']=='Cottage', 'ppt_condensed'] = 'house'
    listd.loc[listd['property_type']=='Villa', 'ppt_condensed'] = 'house'
    listd.loc[listd['property_type']=='Other', 'ppt_condensed'] = 'other'
    listd.loc[listd['property_type']=='Loft', 'ppt_condensed'] = 'room'

    listd=create_dummies(listd,'ppt_condensed')
    listd=create_dummies(listd,'zipcode')

    #imputing some more missing values
    listd['cleaning_fee'].fillna(0, inplace=True)
    listd['beds'].fillna(0, inplace=True)
    listd['bathrooms'].fillna(0, inplace=True)

    #Imputed the rows with price is equal to zero
    listd=listd[listd.price>0]

    listd_drop_cols2=['id', 'listing_url', 'name', 'summary', 'space', 'description',\
       'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',\
       'house_rules', 'picture_url', 'host_id', 'host_url', 'host_since',\
       'host_about', 'host_response_time', 'host_response_rate',\
       'host_acceptance_rate', 'host_thumbnail_url',\
       'host_picture_url', 'host_neighbourhood', 'host_listings_count',\
       'host_total_listings_count', 'host_verifications',\
       'host_has_profile_pic', 'street',\
       'neighbourhood', 'neighbourhood_cleansed', 'city', 'state',\
       'market', 'smart_location', 'country_code', 'country', 'latitude',\
       'longitude', 'is_location_exact', 'property_type',\
       'amenities',\
       'weekly_price', 'monthly_price', 'security_deposit',\
       'extra_people', 'minimum_nights', 'calendar_updated',\
       'has_availability', 'availability_30', 'availability_60',\
       'availability_90', 'availability_365', 'calendar_last_scraped',\
       'number_of_reviews', 'first_review', 'last_review',\
       'review_scores_rating', 'review_scores_accuracy',\
       'review_scores_cleanliness', 'review_scores_checkin',\
       'review_scores_communication', 'review_scores_location',\
       'review_scores_value', 'requires_license', 'license',\
       'jurisdiction_names', 'instant_bookable', 'cancellation_policy',\
       'require_guest_profile_picture', 'require_guest_phone_verification',\
       'calculated_host_listings_count', 'reviews_per_month']
    listd.drop(listd_drop_cols2,axis=1,inplace=True )

    listd['host_is_superhost'] = (listd['host_is_superhost'] == True).astype(int)
    listd['host_identity_verified'] = (listd['host_identity_verified'] == True).astype(int)
    listd['bed_type'] = (listd['bed_type'] == True).astype(int)

    print(listd.shape)
    return(listd)

def create_database(dfclean):
    print('\nCREATING A CLEAN TRAIN TEST DATASET .................')
    print(dfclean.shape)
    print(dfclean.head())

    y= dfclean.price
    X= dfclean.copy()
    X=X.drop(['price'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    X_train.to_pickle('pklz/rand_split/X_train.pkl')
    y_train.to_pickle('pklz/rand_split/y_train.pkl')
    X_test.to_pickle('pklz/rand_split/X_test.pkl')
    y_test.to_pickle('pklz/rand_split/y_test.pkl')

    dfclean_lt_500=dfclean[dfclean.price<500]
    dfclean_gt_500=dfclean[dfclean.price>500]

    y=dfclean_lt_500.price
    X=dfclean_lt_500.copy()
    X=X.drop(['price'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

    X_train.to_pickle('pklz/price_split/X_lt_train.pkl')
    y_train.to_pickle('pklz/price_split/y_lt_train.pkl')
    X_test.to_pickle('pklz/price_split/X_lt_test.pkl')
    y_test.to_pickle('pklz/price_split/y_lt_test.pkl')


    y=dfclean_gt_500.price
    X=dfclean_gt_500.copy()
    X=X.drop(['price'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

    X_train.to_pickle('pklz/price_split/X_gt_train.pkl')
    y_train.to_pickle('pklz/price_split/y_gt_train.pkl')
    X_test.to_pickle('pklz/price_split/X_gt_test.pkl')
    y_test.to_pickle('pklz/price_split/y_gt_test.pkl')


if __name__ == '__main__':
    init()
    df = read_data()
    df = process_data(df)
    create_database(df)