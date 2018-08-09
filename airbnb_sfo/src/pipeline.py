import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error, r2_score
from collections import Counter

from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression

from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.api as sm



def price_xform(dframe,colname):
    dframe[colname]=dframe[colname].str.replace('$','')
    dframe[colname]=dframe[colname].str.replace(',','')
    dframe[colname]=pd.to_numeric(dframe[colname])
    #return(dframe)

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
    dfclean_gt_500=dfclean_gt_500[dfclean_gt_500.price<1500]

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


def regr_plot(y_act,y_pred,mdl_data):
    #Building Residual DF

    dfpred= y_act.to_frame()
    dfpred['preds'] = y_pred
    dfpred['resid'] = dfpred.preds-dfpred.price
    dfpred['residpct'] = (dfpred.preds-dfpred.price)/dfpred.price
    # dfpred.head()

    fig = plt.figure(figsize=(18,5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    ax1.scatter(y_act, y_pred,  color='black')
    ax1.set_title('act_vs_pred')
    ax1.set_xlabel('ACT-PRICE')
    ax1.set_ylabel('PREDICT')

    ax2.scatter(dfpred.price,dfpred.resid, color='red')
    ax2.set_title('RESIDUALS')
    ax2.set_xlabel('PRICE')
    ax2.set_ylabel('RESIDUAL')

    ax3.scatter(dfpred.price,dfpred.residpct, color='blue')
    ax3.set_title('RESIDUAL PCT')
    ax3.set_xlabel('PRICE')
    ax3.set_ylabel('RESIDUAL PCT')

    fig.suptitle('mdl_data', fontsize=12, y=1.2)
    fig.tight_layout()

    fig1 = plt.gcf()
    fig1.savefig('plots/'+'RESI_'+mdl_data+'.png', format='png')

def plot_feature_importance(featurelist,featureimp,name):
    tmp_df=featurelist.to_frame(index=False)
    tmp_df=tmp_df.rename(columns={0:'feature'})
    tmp_df['prime']=featureimp
    feature_df=tmp_df.sort_values(by=['prime'], ascending=True)
    fig=plt.figure(figsize=(10,15))
    ax1=fig.add_subplot(111)
    ax1.barh(feature_df['feature'],feature_df['prime'],color='rgbkymc')
    fig.suptitle(name, fontsize=12, y=1.2)
    fig.tight_layout()
    fig1 = plt.gcf()
    fig1.savefig('plots/'+'FI_'+name+'.png', format='png')


def xtra():

    # X_train= pd.read_pickle('pklz/price_split/X_gt_train.pkl')
    # y_train= pd.read_pickle('pklz/price_split/y_gt_train.pkl')
    # X_test= pd.read_pickle('pklz/price_split/X_gt_test.pkl')
    # y_test= pd.read_pickle('pklz/price_split/y_gt_test.pkl')    
    # X_train=sm.add_constant(X_train,has_constant='add')
    # X_test=sm.add_constant(X_test,has_constant='add')
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # print(X_train.head().T)
    # print(X_test.head().T)

    # listd_insig_cols = ['host_is_superhost','host_identity_verified','bed_type','guests_included','room_type_full',\
    #                 'room_type_pvt','amentcnt','ppt_condensed_apt','ppt_condensed_aptspl','ppt_condensed_auto',\
    #                 'ppt_condensed_hotel','ppt_condensed_other',\
                    # 'zipcode_94014.0', 'zipcode_94015.0', 'zipcode_94102.0',\
                    # 'zipcode_94103.0', 'zipcode_94104.0', 'zipcode_94105.0',\
                    # 'zipcode_94107.0', 'zipcode_94108.0', 'zipcode_94109.0',\
                    # 'zipcode_94110.0', 'zipcode_94111.0', 'zipcode_94112.0',\
                    # 'zipcode_94114.0', 'zipcode_94115.0', 'zipcode_94116.0',\
                    # 'zipcode_94117.0', 'zipcode_94118.0', 'zipcode_94121.0',\
                    # 'zipcode_94122.0', 'zipcode_94123.0', 'zipcode_94124.0',\
                    # 'zipcode_94127.0', 'zipcode_94129.0', 'zipcode_94131.0',\
                    # 'zipcode_94132.0', 'zipcode_94133.0', 'zipcode_94134.0','zipcode_94158.0']
    # listd_insig_cols = ['zipcode_94014.0', 'zipcode_94102.0',\
    #                 'zipcode_94103.0', 'zipcode_94104.0', 'zipcode_94105.0',\
    #                 'zipcode_94107.0', 'zipcode_94108.0', 'zipcode_94109.0',\
    #                 'zipcode_94110.0', 'zipcode_94112.0',\
    #                 'zipcode_94114.0', 'zipcode_94115.0',\
    #                 'zipcode_94015.0', 'zipcode_94111.0', 'zipcode_94116.0',\
    #                 'zipcode_94117.0', 'zipcode_94118.0', 'zipcode_94121.0',\
    #                 'zipcode_94122.0', 'zipcode_94123.0', 'zipcode_94124.0',\
    #                 'zipcode_94127.0', 'zipcode_94129.0', 'zipcode_94131.0',\
    #                 'zipcode_94132.0', 'zipcode_94133.0', 'zipcode_94134.0','zipcode_94158.0',\
    #                 'ppt_condensed_apt','ppt_condensed_aptspl','ppt_condensed_auto',\
    #                 'ppt_condensed_hotel','ppt_condensed_other','ppt_condensed_hostel','ppt_condensed_house',\
    #                 'min_night_stay_long','min_night_stay_mid']
    # X_train.drop(listd_insig_cols,axis=1,inplace=True)
    # X_test.drop(listd_insig_cols,axis=1,inplace=True)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


    # modelOLS = sm.OLS(y_train, X_train)
    # resultsOLS = modelOLS.fit()
    # print(resultsOLS.summary())
    # y_predOLS = resultsOLS.predict(X_test)
    # my_metrics(y_test,y_predOLS,'OLS_LinReg_gt500_Dataset')
    # regr_plot(y_test,y_predOLS,'OLS_LinReg_gt500_Dataset')


    # #SK Learn Linreg
    # lmodel = linear_model.LinearRegression()
    # lmodel.fit(X_train,y_train)
    # y_predsk = lmodel.predict(X_test)
    # print('Coefficients: \n', lmodel.coef_)
    # my_metrics(y_test,y_predsk,'SKLearn Linear Reg Metrics')
    pass

def my_metrics(y_act,y_pred,mdl_data):
    r2= (1 - (((y_act-y_pred)**2).sum()) / (((y_act-y_act.mean())**2).sum()) )
    r2_score= (1 - ((y_act-y_pred)**2).sum() / ((y_act-y_act.mean())**2).sum() )
    mse =  ((y_act-y_pred) ** 2).sum()/len(y_act)
    rmse = mse ** 0.5
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_act))**2))
    mspct = np.abs(((y_act-y_pred)/y_act)).mean()


    mse_prop = (np.mean(np.abs((y_act - y_pred)**2 / (y_act+1)**2  )) * 100) **0.5

    #Calculate the baseline errors
    baseline = abs(y_act.mean()-y_act)
    #Calculate the absolute errors
    residuals = abs(y_pred-y_test)
    #Calculate Mean Absolute pct error
    mape = (residuals/y_test)*100
    #Accuracy 
    accy=100-np.mean(mape)
    #Mean Absolute Error 
    
    print('\n**************** Metrics for : ',mdl_data,' *******************')
    print("   R2 (Variance Squared ERROR)                   = %.4f"%r2)
    print("   MSE (Mean Squared ERROR)                      = %.4f"%mse)
    print("   RMSE (ROOT Mean Squared ERROR)                = %.4f"%rmse)
    print("   RMSLE (ROOT Mean Squared Logrithmic ERROR)    = %.4f"%rmsle)
    print("   PCTE  (Percent Absolute Error)                = %.4f\n"%mspct)

    print("   Baseline Mean Absolute Error                  =",np.mean(baseline))
    print("   Predicted Mean Absolute Error                 =",np.mean(residuals))
    print("   Accuracy  (100-MAPE mean absolute PCT ERR)    =",round(accy,2), '%')

    print('\n**************** -------------------------- *******************',len(y_act), mse_prop)
    # print(len(y_act), mse_prop)

    # regr_plot(y_act,y_pred,mdl_data)
    return(r2,rmse,rmsle,mspct)

def run_randforest(X_train,y_train,X_test,y_test,name):
    print('\nRUNNING RANDOM FOREST .................  :',name)

    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

    print(regr.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    y_predRF=regr.predict(X_test)
    my_metrics(y_test,y_predRF,name)

    featurelist=X_train.columns
    featureimp= regr.feature_importances_
    plot_feature_importance(featurelist,featureimp,name)
    regr_plot(y_test,y_predRF,name)




def run_gradientboost(X_train,y_train,X_test,y_test,name):
    print('\nRUNNING GRADIENT BOOST REGRESSION .................  :',name)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,\
          'learning_rate': 0.01, 'loss': 'ls', 'random_state':0}

    gbr = GradientBoostingRegressor(**params)
    # gbr = GradientBoostingRegressor(max_depth=2, random_state=0)
    gbr.fit(X_train, y_train)

    y_predGBR = gbr.predict(X_test)
    mse= mean_squared_error(y_test,y_predGBR)
    print("MSE: %.4f" %mse)
    # print("My GBR Score : %.4f",my_metric(y_test,y_predGBR))

    print(gbr.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    y_pred=gbr.predict(X_test)
    my_metrics(y_test,y_pred,name)

    featurelist=X_train.columns
    featureimp= gbr.feature_importances_
    plot_feature_importance(featurelist,featureimp,name)
    regr_plot(y_test,y_pred,name)





def run_linreg(X_train,y_train,X_test,y_test,name):
    print('\nRUNNING LINEAR REGRESSION .................  : ',name)   
    X_train=sm.add_constant(X_train,has_constant='add')
    X_test=sm.add_constant(X_test,has_constant='add')
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # print(X_train.head().T)
    # print(X_test.head().T)

    # Statmodel Linreg
    modelOLS = sm.OLS(y_train, X_train)
    resultsOLS = modelOLS.fit()
    print(resultsOLS.summary())
    y_predOLS = resultsOLS.predict(X_test)
    my_metrics(y_test,y_predOLS,name)
    regr_plot(y_test,y_predOLS,name)



if __name__ == '__main__':
    init()

    # df = read_data()
    # df = process_data(df)
    # create_database(df)

    X_train= pd.read_pickle('pklz/price_split/X_lt_train.pkl')
    y_train= pd.read_pickle('pklz/price_split/y_lt_train.pkl')
    X_test= pd.read_pickle('pklz/price_split/X_lt_test.pkl')
    y_test= pd.read_pickle('pklz/price_split/y_lt_test.pkl') 

    run_linreg(X_train,y_train,X_test,y_test,'OLS_LinReg_lt500_Dataset')
    run_randforest(X_train,y_train,X_test,y_test,'RF_lt500_Dataset')
    run_gradientboost(X_train,y_train,X_test,y_test,'GB_lt500_Dataset')