This is an instance-based self-adaptive scale fusion implementation built on PyTorch.
In ./model/adaptiveScaleFusion.py, hybrid weights are assigned across two instance scales, and the method has been evaluated on abmil, clam, transmil, acmil, and other models.
You can run it directly with

`python main.py --config 'tcga_multi_weighted' --source 'ctrans' --alpha 0.5 --concat 'concat' --model 'acmil' --n_tokens 5 --gpu '0'`
