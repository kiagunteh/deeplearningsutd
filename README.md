# Malicious Packet Neural Network Classifier
Training a deep learning model  to classify anomalous malicious network packets.\
For the module 50.039 Deep Learning, Y2026 @ SUTD
## About
The project focuses on anomaly detection, with the requirement to use a dataset that exhibits class imbalance, where the number of "normal" samples far exceeding the number of "anomalous" samples. The task we have chosen is the detection of malicious network packets, with the dataset being the UNSW_NB15 dataset found [here](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data).\
\
Details about data preprocessing and the deeplearning architecture used and its implementation can be found within the PDF report and Jupyter notebook.
## Project Structure
```
.
├── helper_functions
│   ├── model.py
│   ├── preprocessing.py
│   ├── training.py
│   └── visualisation.py
├── saved_models
|   └── model_before_tuning.pth
├── tuning
|   ├── best_model.pth
|   └── best_params.json
├── .gitignore
├── project.ipynb
├── requirements.txt
├── tuning.db
├── tuning.py
└── README.md
```

| No | File Name | Details 
|----|------------|-------|
| 1  | project.ipynb | Main project file
| 2  | helper_functions | Contains helper functions imported by project.ipynb, split by functionality into their individual files
| 3  | saved_models | Contains trained model file used by project.ipynb for reproducibility
| 4  | tuning | Contains tuned model file and best parameters in a JSON file, used by project.ipynb for reproducibility
| 5  | tuning.py | Contains code to conduct hyperparameter tuning using Optuna
| 6  | download_dataset.py | Contains code to download and upzip the dataset to be placed in the correct ./data directory for project.ipynb

## Setting Up
### 1. Clone Repository
```bash
$ git clone https://github.com/kiagunteh/deeplearningsutd
$ cd deeplearningsutd
```
### 2. Install Dependencies
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```
### 3. (Optional) Download and install cuda
Our model will use cuda to run the model on your GPU if you install cuda, which will accelerate training and evaluation. However, this is entirely optional and the notebook will stil run without cuda on your CPU instead.\
First check if your GPU is supported by checking this [link](https://developer.nvidia.com/cuda/gpus).
#### Within WSL
You may follow the steps detailed within this [link](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to get cuda working within your WSL environment.
#### Not WSL
Follow the steps [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local) to get cuda on your machine.
### 4. Download Dataset
```bash
$ python3 download_dataset.py
```
## Running the Jupyter notebook
We used the [Jupyter VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter&ssr=false#overview) to run and interact with our notebook. Simply install it within VSCode and open the Jupyter notebook with VSCode to start.
