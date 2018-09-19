# README 

This repo contains Python implementations for data science methods most pertinent to life insurance industry use-cases. 
These include various binary classification, multiclass classification, and regularized regression examples and implementations. 
Furthermore, a module `textual_data_handling.py` contains implementation for Latent Dirichelet Allocation -- encapsulated in a classifier that allows for training() and classifying() -- that is essential for handling any sort of textual data (i.e. medical keyword data / descriptions). 

This repo also contains two other projects associated with asset management. Due to privacy concerns, we only host the publicaly-available R and Powershell source code here. These projects are: 
  * peer group portfolio analysis
  * ARIMA-based forecasting of mortgage-backed security "health data"
  * Basic Powershell tutorial 
  * Using Powershell to aggregate KIC tables 

All implementations require Python 3+, R-Studio, base Windows Powershell (respectively)

## Setup 
Data science files require the Anaconda Distribution. Install Anaconda Navigator on Windows
Asset management files require R Studio and associated package dependencies listed in the .rmd file

## Package Dependencies 
#### Python 
 - scikit-learn 
 - numpy 
 - pandas 
 - pathlib 
 - gensim
 - nltk 
#### R
 - ggplot 
 - dplyr
 - treemapify 
 - knitr 
 - readxl 
 - gridExtra 
#### Powershell 
 - ImportExcel 
 - PSExcel 

For Python: Install via opening anaconda navigator --> Environments -> base root -> Open Terminal -> Enter "conda install *packagename*
  
  
  
## Description of Python Files 
Please refer to docstrings for each `.py` file in Python_Files folder 




