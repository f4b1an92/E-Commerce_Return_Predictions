# Return Predictions in E-Commerce

## Abstract

Key words: E-Commerce, Ensemble Learning


## 0. Preliminaries
To execute the provided code, just follow the steps outlined below:

```
1.  Open a terminal and navigate to the location of the project folder.

2.  Now set up a virtual environment using venv
    -   py -m pip install --user virtualenv		# installs venv 
    -   py -m venv env					# creates virtual environment

3.  Activate the environment:
    -   cd env\Scripts
    -   activate

4.  Go back to the main project directory

5.  Ensure that your Python version is 3.8.8 
    -   py --version

6.  Install dependencies from requirements.txt 
    -   pip install -r requirements.txt

7.  Execute the main script which will go through the code to reproduce the experiments itself.
    -   py main.py
```

## 1. Data 
The data set consists of two separated files ("known" & "unknown"), one with 100,000 data points and 14 variables ("known"-data set) & one with 50,000 
data points and 13 variables ("unknown"). The missing variable in the "unknown"-set is the binary target variable "return". Therefore, the "unknown"-data
set will only be used for prediction and it's evaluation will be conducted via the corresponding [Kaggle-competition](). Subsequently, the "known"-data 
set will be used for training and testing the model.

The data sets can be retrieved from here: 

If you want to run the experiments, please create a folder called "data" in the repositories directory and download the data files from the above 
URL into the newly created "data"-folder.


## 2. Exploratory Data Analysis


## 3. Feature Engineering & Selection


## 4. Modelling  


## 5. Conclusion


## 6. References

