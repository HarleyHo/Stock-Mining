# Stock-Mining
This project is trying to use some data mining techniques to find some interesting patterns in Hang Seng Index (HSI) and Hong Kong stock market.

This is a course project of CS5483 at City University of Hong Kong.

## Data
The data is gathered by [TuShare](https://tushare.pro/). We get the data by their API and save them into csv files.

We obtain the data of HSI from 12/04/2015 to 12/04/2025, and the data of Hong Kong stock market from 12/11/2024 to 12/04/2025.
The obtained data is well processed, and we only need to extract some features we need from the data.

## Methods

## Usage
First you need to install the dependencies with ```pip install -r requirements.txt```.

Then you can run ```python dataset_obtaining.py``` to gain the data.

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
