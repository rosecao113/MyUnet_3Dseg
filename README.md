# MyUnetPipeline
This is the very first unet pipeline from scratch

the training image and labels are nifti files, mhd files could be transformed to nifti files first.

$ python unet_training.py

the models would be saved in ./runs_dict

$ python unet_evaluation.py

the output masks would be saved in ./tempdir


