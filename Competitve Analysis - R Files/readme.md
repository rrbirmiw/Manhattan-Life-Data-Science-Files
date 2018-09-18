#README 

The main file in this folder is “ARIMA Forecast Paper.” The purpose of this paper was to analyze all structured/mortgage CUSIPS in the portfolios and identify potential warning signs regarding delinquencies/CPR and principal pay-downs. 

The method used to model and analyze was the ARIMA method. ARIMA is a standard and widely-used family of methods for statistical time-series modelling and forecasting.  

## Abstract
The “health” of a mortgage-backed security/bond (RMBS) can be measured by certain rates (“speeds”) of its underlying components. 
Some examples of these rates/speeds include 60-Day Delinquency Rate, CPR (conditional prepayment rate), etc. 
For example, RMBS’s whose underlying delinquency rates are increasing, posit risks to the bondholder, assuming trends continue. 
However, these rates/speeds do not necessarily follow simple linear/polynomial trends: 
it is well-known that e.g. the housing market is rather cyclical and therefore forecasting future rates can be rather difficult. 
In this paper we utilize ARIMA time-series forecasting to accurately forecast these rates and identify potential “worrisome” 
CUSIPS within the portfolio. In particular, we use ARIMA to project outward 12 months, 
then rank the CUSIPS according to year-over-year forecasted percent changes. 


## Code Usage
The R file containing the source code is called arima_analysis_report.rmd. This module requires the files cpr.xlsx and dq.xlsx (both provided in this project’s subfolder). Hitting ‘Run All’ in R will automatically generate the csv file with the forecasted data, as well as two example ways to view an individual cusip in more detail. The latter can be executed by running the line, in R: 
To view CPR forecasts, for cusip “foo”: `main("foo", 'cpr.xlsx', TRUE)`
To view DQ forecasts, for cusip “foo”: `main("foo", 'dq.xlsx', TRUE)`



