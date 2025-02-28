# ECCL
The is the official repository with Pytorch version for [Efficient Cross-Modal Semantic Correspondence Learning for Text-to-Image Person Retrieval]
![image](https://github.com/user-attachments/assets/ef55e733-f0c5-41db-ac89-afe7a6368e59)

## Installation
- pytorch 
- torchvision
- prettytable
- easydict
- loralib


## Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training
```
sh run.sh
```

## Testing
```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```
