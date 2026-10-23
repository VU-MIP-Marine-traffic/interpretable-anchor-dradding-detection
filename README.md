## Set-up

Running this code requires two different virtual environments.
This is because some code needed to be run on HPC and it uses older Python and tensorflow versions.

#### To set-up the environment `venv` do the following:
1. Install Python 3.8.5
2. Using it create a virtual environment

Windows:
`python -m venv venv`

Linux (if not Debian-based distro, meaning the following doesn't work, try the previous command):
`python3 -m venv venv`

3. Install the necessary libraries: 

Windows:
```
./venv/bin/activate
pip install -r venv-requirements.txt
```

Linux:
```
source venv/bin/activate
pip install -r venv-requirements.txt
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

```bash
API_KEY="<api-key>"
```

3. The AIS data for ships around Yi-Peng 3 case can be obtained
4. If you get the key and put everything correctly in step 2: you can get meteorological data.

## How to run this repository/Repository structure




## AI usage

AI was used to clean up the repository.
The chats used for that can be viewed:
1. https://claude.ai/share/5fe4b4a4-e910-430f-966f-a8e3e89c281f
2. https://chatgpt.com/share/6acd1a99-96d8-83eb-a911-4f552c088d5b
3. https://claude.ai/share/21ad5497-3ab4-4fb9-9c18-b130143181d8