# Stock-Mining
This project is trying to use some data mining techniques to find some interesting patterns in Hang Seng Index (HSI) and Hong Kong stock market.

This is a course project of CS5483 at City University of Hong Kong.

## Data
The data is gathered by [TuShare](https://tushare.pro/). We get the data by their API and save them into csv files.

We obtain the data of HSI from 12/04/2015 to 12/04/2025, and the data of Hong Kong stock market from 12/11/2024 to 12/04/2025.
The obtained data is well processed, and we only need to extract some features we need from the data.

## Methods
We use some data mining techniques to find some interesting patterns in the data.
### Clustering
We use K-means and hierarchical clustering to cluster the data into some clusters.

For HSI, we try to analyse the relationship between the time and the index fluctuation,
so we use K-means to find some patterns.

For those stocks, we use hierarchical clustering to find some patterns, trying to use an easy way to find some stocks performing similarly.
Different people may have different opinions on what is a good stock, or they have different ability of bearing risk, then they may choose a set of stocks that matches them.

### Association Rule
We use Apriori algorithm to find some association rules between the features and the fluctuation of HSI.
This can help us find some important features that affect the performance of the index, which can help us to make a better choice when we need to use some features to predict the index.

### Decision Tree
Decision Tree is used to offer an easier and more effective way to predict the increase / decrease of the HSI.
Although the accuracy may not be very high compared to methods like time series or Long Short Term Memory (LSTM), it is still useful given that it is a simple and effective way to predict the index.

## Usage
First you need to install the dependencies with ```pip install -r requirements.txt```.

Then you can run ```python dataset_obtaining.py``` to obtain the data.

After that, you can run other python codes to get the related results.

## Results
The results are shown in the [report](report.pdf).

## Dependencies
- [X] Pandas
- [X] Numpy
- [X] Matplotlib
- [X] Seaborn
- [X] Sklearn
- [X] Scipy
- [X] Mlxtend