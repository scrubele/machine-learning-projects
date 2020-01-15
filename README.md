# Digits sounds recognizer
To run code:
```
pip3 install -r requirements.txt
python3 digits_sounds_recognizer.py
```
<i>In order to run in the Logging mode, change value of LOG_MODE in the config.py</i>

Files structure:
```
├── digits_recognizer
│   ├── digits_sounds_recognizer.py #main file
│   ├── config.py   # config file
│   ├── file_adapters 
│   │   ├── files_adapter.py
│   │   └── wav_processor.py
│   ├── model_services 
│   │   ├── log_generator.py
│   │   └── results_evaluator.py
│   └── utilities
│       ├── array_processor.py
│       └── graphs_plotter.py
│   ├── model.h5 # model
│   ├── model.json # jsonable model
│   ├── history.json # loss and accuracy result values (used for the plots) 
├── logs
│   └── accuracy.png
├── recordings
│   ├── all_recordings [2000 entries exceeds filelimit, not opening dir] # FSDD dataset
│   └── test_records  # all test records 
└── tree
```
Used dataset:
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
 
 
