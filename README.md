# Multivariate time series prediction for price of Bitcoin using LSTM

This project intends to serve as a starting point to explore multivariate time series prediction for predicting the closing price of Bitcoin (BTC), using LSTM network. Apart from considering only the trading features of BTC, the project goes further to consider the influence of certain other commodities, stocks and index (ex. Gold, Oil, S&P500, USDX etc.) on BTC price. This opens up a multitude of possibilities to further analyze and experiment with this baseline. From the feature engineering and modeling point of view, some ideas to take this project further could be:

1. Analyze correlation vs causation (Check correlation graph for different timelines, ex. data after 2019 vs data after 2021) among considered features and select most effective/contributory. Sticking to only BTC features, or adding features apart from the ones already included, can also be a consideration!
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
- `.gitignore`
- `main.py`
- `pre_process.py`
- `requirements.txt`

Upon project directory setup, the execution can be done using the following 2 steps. Make sure to change the data paths in the respective files accordingly before execution.
```bash
python pre_process.py
```
```bash
python main.py
```

Post model training and inference, a plot visualizing model predictions vs original BTC price will be generated:

<img width="815" alt="BTC price plot_pred vs org 3" src="https://github.com/VIJVIV/Multi_BTC_Pred/assets/146338220/c83c8c9e-37dd-4753-88ec-490930edffa7">



