# Shallow Triple Stream CNN (S3S-CNN) for Micro-Expression Recognition

## Setup and Installing Dependencies
```
# base conda environment
$ conda install -c conda-forge mamba

# S3S-CNN environment
$ mamba env create --file environment.yml
```

## Generating Features
* Go inside the `features` folder.
```
$ cd features
```
* Modify `config.json`.
```
{
    "emotions": ["anger", "sadness", "surprise", "fear", "happiness", "disgust"],
    "dbs": ["SAMM", "MMEW", "CASME_II"],
    "db_OF": ["SAMM", "CASME_II"],
    "img_dim": [128, 128],
    "data_dir": "path to raw dataset",
    "out_dir": "path to save features extracted",
    "facs_dir": {
        "SAMM": "path/to/SAMM_FACS.csv",
        "MMEW": "path/to/MMEW_FACS.csv",
        "CASME_II": "path/to/CASME_II_FACS.csv"
    },
    "raw_file": "path/to/onset_apex.json",
    "dataset": "path/to/dataset.json",
    "face_bounding_box": "path/to/face_bounding_box.json"
}
```
* Run `data_util.py` to extract and generate optical flow features.
```
$ python data_util.py
```

## Training Model
* Go inside the `MER` folder.
```
$ cd MER
```
* Modify `mer_config.json`.
```
{
    "lr": 0.001,
    "batch_size": 128,
    "epoch": 30,
    "img_dim": 128
}
```
* Run `main.py` to train the model.
```
$ python main.py
```