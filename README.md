## Set-up

Running this code requires two different virtual environments.
This is because some code needed to be run on HPC and it uses older Python and tensorflow versions.

#### To set-up the environment `venv` do the following:
1. Install Python 3.8.5
2. Using it create a virtual environment:
```
python -m venv venv      # If using Linux (debian-based distro) use python3
python -m venv qrf-venv  # If using Linux (debian-based distro) use python3
```
3. Install the necessary libraries: 

Windows:
```
./venv/bin/activate
pip install -r venv-requirements.txt
deactivate
./qrf-venv/bin/activate
pip install -r qrf-venv-requirements.txt
```

Linux:
```
source venv/bin/activate
pip install -r venv-requirements.txt
deactivate
source qrf-venv/bin/activate
pip install -r qrf-venv-requirements.txt
```

#### To set up the environment `qrf-venv` follow the following steps:

1. Install Python 3.12.1
2. Follow steps 2-3 in previous section but **everywhere** instead ov `venv` put `qrf-venv`

Windows:
```
python -m venv venv
./venv/bin/activate
pip install -r venv-requirements.txt
```

Linux:
```
python3 -m venv venv
source venv/bin/activate
pip install -r venv-requirements.txt
```

#### Data availability:

1. Yi-Peng 3, VEZHEN and EAGLE S data is unavailable.
2. Meteorological data also needs to be collected, for that I cannot provide the data or the API key, so please get an API key from stormglass and put it into a .env file like so:

```
API_KEY="<api-key>"
```

3. The AIS data for ships around Yi-Peng 3 case can be obtained
4. If you get the key and put everything correctly in step 2: you can get meteorological data.

## How to run this repository/Repository structure

When you run the code, data (graphs and figures) in the paper will appear in the terminal and output/ folder.

0. Put the data files for EAGLE S, VEZHEN and ships around them in data/.
1. To get the data for vessels around YI-PENG 3, run src/data_collection.py.
2. To prepare the datasets EV and YP3 as well as the anomaly prepared datasets, use:
```
# with venv activated
python -m src.yipeng3.data_preparation_normal lstm-ae
python -m src.eagles_vezhen.data_preparation_eagles lstm-ae
python -m src.yipeng3.data_preparation_anomalies lstm-ae
python -m src.eagles_vezhen.data_preparation_anomalies lstm-ae

# with qrf-venv activated
python -m src.yipeng3.data_preparation_normal qrf
python -m src.eagles_vezhen.data_preparation_eagles qrf
python -m src.yipeng3.data_preparation_anomalies qrf
python -m src.eagles_vezhen.data_preparation_anomalies qrf
```
This will create all of the numpy files needed in output/data_{eagles_vezhen or yipeng3}.

3. To get EDA results:
```
cd notebooks/eda
python anomalies-eagle-s-vezhen-20260617.py
python vezhen-storm-check.py
python whole-dataset-vessels-around-anom.py
cd ../..
```
4. To optimize hyperparameters for LSTM-AE and do its experiments run:
```
# with venv activated
python -m src.eagles_vezhen.lstm-ae-hyperparameter-tuning
python -m src.yipeng3.lstm-ae-hyperparameter-tuning
python -m src.eagles_vezhen.lstm-metrics
python -m src.yipeng3.lstm-metrics
```
5. To optimize QRF model and run its experiments run:
```
# with qrf-venv activated
python -m src.eagles_vezhen.qrf-tree-selection
python -m src.eagles_vezhen.qrf-hyperparameter-selection
python -m src.eagles_vezhen.qrf-metrics

python -m src.yipeng3.qrf-tree-selection
python -m src.yipeng3.qrf-hyperparameter-tuning
python -m src.yipeng3.qrf-metrics
python -m src.yipeng3.qrf-res-yipeng3

cd notebooks/modelling
python qrf-tree-count.py
python qrf-yp3-tree-count.py
python qrf-eagles-vezhen.py
cd ../..
```
6. To optimize XGBoost and run its experiments:
```
python -m src.eagles_vezhen.xgboost-hyperparam-selection
python -m src.eagles_vezhen.xgb-metrics
python -m src.yipeng3.xgboost-hyperparam-selection
python -m src.yipeng3.xgb-metrics

cd notebooks/modelling
python xgboost-hyperparams-yipeng3.py
python xgboost-hyperparams.py
python xgboost-ev-validation-explanations.py
python xgboost-yp3-validation-explanations.py
python xgboost-abblation.py
cd ../..
```
7. For comparisson of all three models run:
```
python -m notebooks.modelling.metrics-comparisson
```

## AI usage

AI was used to clean up the repository.
The chats used for that can be viewed:
1. https://claude.ai/share/5fe4b4a4-e910-430f-966f-a8e3e89c281f
2. https://chatgpt.com/share/6acd1a99-96d8-83eb-a911-4f552c088d5b
3. https://claude.ai/share/21ad5497-3ab4-4fb9-9c18-b130143181d8
