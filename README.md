# ECCL
The is the official repository with Pytorch version for [Efficient Cross-Modal Semantic Correspondence Learning for Text-to-Image Person Retrieval]
![image](https://github.com/user-attachments/assets/ef55e733-f0c5-41db-ac89-afe7a6368e59)

## Installation
- pytorch 
- torchvision
- prettytable
- easydict
- loralib (Replace the corresponding file in loralib with the layers.py from this repository.)


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

## Main results
### CUHK-PEDES dataset

|     Method      |     Backbone     |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |
| :-------------: | :--------------: | :-------: | :-------: | :-------: | :-------: | 
|     CMPM/C      |    RN50/LSTM     |   49.37   |     -     |   79.27   |     -     | 
|      DSSL       |    RN50/BERT     |   59.98   |   80.41   |   87.56   |     -     | 
|      SSAN       |    RN50/LSTM     |   61.37   |   80.15   |   86.73   |     -     | 
|   Han et al.    |  RN101/Xformer   |   64.08   |   81.73   |   88.19   |   60.08   | 
|      LGUR       | DeiT-Small/BERT  |   65.25   |   83.12   |   89.00   |     -     |
|       IVT       |  ViT-B-16/BERT   |   65.59   |   83.11   |   89.21   |     -     |
|      CFine      |  ViT-B-16/BERT   |   69.57   |   85.93   |   91.15   |     -     |
| IRRA | ViT-B-16/Xformer | 73.38 | 89.93 | 93.71 | 66.13 |
|     RaSa    | ViT-B-16/Bert-Base |   76.51   |   90.29   |   94.25   |   69.38   |
|     MLLM4Text-ReID    | ViT-B-16/Xformer |   76.82   |   **91.16**   |   **94.46**   |   69.55   |
| **ECCL (ours)** | ViT-B-16/Xformer |**77.17** | 90.64 | 94.39 | **69.81** | 

### ICFG-PEDES dataset
|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    | 
| :-------------: | :-------: | :-------: | :-------: | :-------: |
|     CMPM/C      |   43.51   |   65.44   |   74.26   |     -     |
|      SSAN       |   54.23   |   72.63   |   79.53   |     -     | 
|       IVT       |   56.04   |   73.60   |   80.22   |     -     |
|      CFine      |   60.83   |   76.55   |   82.42   |     -     | 
| IRRA | 63.46 | 80.24 | 85.82 | 38.05 | 7.92 |
|    RaSa     |   65.28   |   80.40   |   85.12   |   41.29   | 
|      MLLM4Text-ReID      |   67.05   |   82.16   |   **87.33**   |     41.51     | 
| **ECCL (ours)** | **68.03** | **82.46** | 87.28 | **43.50** | 

### RSTPReid dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |
| :-------------: | :-------: | :-------: | :-------: | :-------: |
|      DSSL       |   39.05   |   62.60   |   73.95   |     -     |
|      SSAN       |   43.50   |   67.80   |   77.15   |     -     |
|       IVT       |   46.70   |   70.00   |   78.80   |     -     |
|      CFine      |   50.55   |   72.50   |   81.60   |     -     |
| IRRA | 60.20 | 81.30 | 88.20 | 47.17 | 25.28 |
|    RaSa     |   66.90   |   86.50   |   91.35   |   52.31   |
|      MLLM4Text-ReID      |   68.50   |   **87.15**   |   **92.10**   | 
| **ECCL (ours)** | **69.60** | 85.95 | 91.05 | **55.08** |

## Acknowledgement
Our code is conducted based on [IRRA](https://github.com/anosorae/IRRA).
