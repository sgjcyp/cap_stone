# Price Is Right - Price Predictor for Airbnb

### Motivation
Airbnb listings are growing at rapid pace all over the world.  New hosts price their listings approximately based on the listings in the neighbourhood.  Most often than not, it is hard to find the listing with the same features in the same location. This Model helps hosts price their properties @ Right Price based on the primary and latent features

### Data
Source: InsideAirbnb.com

The data set used in this project consists of 5k-8k listings each day. Each listing has inforamation about 96 features.   Each feature can have multiple variations causing it to be a OOC multidimentional model.  For example a listing with "1 bedroom + 1 pvt bathroom" will be priced higher than "1 bedroom + 1 shared bathroom"

### Process Flow 

The data was obtained from Insideairbnb.com.  After deep evaluation and cycles of learning,  final dataset was created for common listings excluding luxury listings based on time series. This model uses all the features to make the prediction of daily price.

### Key Features :
  - Data Engineering
  - Regression Modelling
  - Hyperparameter Tuning
  - Prediction and Result Analysis 

### Challenges & Limitations :
  - Rich dataset with several features with multiple options  
  - Some of the sparse outliers were breaking the model performance.   
  - Excluded the luxury listings, Model is limited to common listings

### Tech used:
Python, Pandas, Numpy, Matplotlib, statsmodels, Scikit-learn, Feature Engg, Modelling, Regression, gmaps
