# Multivariate Bitcoin Price Prediction using LSTM

This project intends to serve as baseline to explore multivariate time series prediction for predicting the closing price of Bitcoin (BTC) using LSTM. Apart from considering only the trading features of BTC, the project goes further to consider the influence of certain other commodities, stocks and index (ex. Gold, Oil, S&P500, USDX etc.). This opens up a multitude of possibilities to further analyze and experiment with this baseline. From the feature engineering and modeling point of view, some ideas to take this project further could be:

1. Explore correlation vs causation (Hint: check correlation graph for different timelines, ex. data after 2018 and data after 2019) among considered features and analyze which features really contribute to better BTC price prediction. Features apart from the ones included can also be a consideration!
2. Optimize training strategy to tackle overfitting/underfitting
3. Experiment and compare with other time series models/architectures 

Project dependencies can be installed using the requirements file
```bash
pip install -r requirements.txt
```

Raw data required for analysis can be directly downloaded from [here](https://drive.google.com/drive/folders/11qirLsWjUPwTzwq6b8Che-L1Mjk1l7e3?usp=sharing). The individual data categories was sourced from [Investing.com](https://www.investing.com/)

The project directory was setup as follows:

#### Project Root
- `raw_data`
    - `BTC Historical Data.csv`
    - ...
    - `USD Index Historical Data.csv`
- `main.py`
- `pre_process.py`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `LICENSE`

Upon having the raw data folder saved under the project root directory, the project execution can be done using the follwing 2 steps. Make sure to change the data paths in the respective files accordingly before execution.
```bash
python pre_process.py
python main.py
```




