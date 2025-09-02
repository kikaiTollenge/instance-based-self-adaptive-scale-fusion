# Instance-based self-adaptive scale fusion for whole slide image classification

This is an instance-based self-adaptive scale fusion implementation built on PyTorch. 


![Fig2](https://github.com/user-attachments/assets/7bd3c718-7b3d-483e-a462-49e151307801)

## Environment
Installation environment dependencies.Specially,torch==2.0.1, openslide==1.1.2

`pip install -r requirements.txt`

## DataProcessing

Whole slide image which endswith '.tif' or '.svs' is available to process with the script ./dataset/dual_sample.py. After installing OpenSlide and modifying the file path and, you can run it directly with

`python ./dataset/dual_sample.py --dataset 'c16'`



## Feature Extract

Feature extract can be run by the script ./ctranspath/get_feature.py. The detail can be found at <[Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)>



## Training

In ./model/adaptiveScaleFusion.py, hybrid weights are assigned across two instance scales, and the method has been evaluated on abmil, clam, transmil, acmil, and other models. You can run it directly with

```
python main.py --config 'tcga_multi_weighted' --source 'ctrans' --alpha 0.5 --concat 'concat' --model 'acmil' --n_tokens 5 --gpu '0'
```

Other experiments with fixed weights can also be run through

`python main.py --config 'tcga_multi_custom' --source 'ctrans' --alpha 0.5 --concat 'concat' --model 'acmil' --n_tokens 5 --gpu '0'`

High Scale Experiment can be run through

`python main.py --config 'tcga_single_high' --source 'ctrans' --alpha 0.5 --concat 'none' --model 'acmil' --n_tokens 5 --gpu '0'`

Low Scale Experiment can be run through


`python main.py --config 'tcga_single_low' --source 'ctrans' --alpha 0.5 --concat 'none' --model 'acmil' --n_tokens 5 --gpu '0'`


