

# Diabetic retinopathy recognition #

## Preparations
Before beginning with the process, we should choose the model and run mode.
<pre>
flags.DEFINE_string('model_id', 'B1', 'Choose your model from the Model index.')
</pre>
Following network architectures are available:
<pre>
Model_index = {'18': 'ResNet18', 
'34': 'ResNet34',  '50': 'ResNet50', '101':'ResNet101', '152': 'ResNet152',
'121': 'DenseNet121', '169': 'DenseNet169', '201': 'DenseNet201', '264': 'DenseNet264',
'IRV2': 'InceptionResnetV2', 
'IV3': 'InceptionV3',
'B2': 'EfficientNetB2', 'B1': 'EfficientNetB1', 'B0': 'EfficientNetB0',
'M2': 'MobilenetV2',
'16': 'vision_transformer_16'}
</pre>
Specify the running mode
<pre>
flags.DEFINE_string('Mode', 'Wandb', 'Specify mode from: Train, Evaluate, Tune/Wandb, Ensemble or Kaggle.')
</pre>
In ```config.gin``` you can set the configurable parameters. In the following modes you can find corresponding configurable parameters in details. 

## Train
There's two kinds of training process can be chosen, namely normal training process and k-fold training process. You can simply set ```KFOLD=True``` in ```main.py``` to start training process of k-fold.

Since the ```input_pipeline.file_extraction.load_file_names``` is a generator of k-fold indexes of each dataset for __k__ iterations, the normal training process only trains the first round of iteration but otherwise all __k__ iterations will be run out in k-fold training process

all the k-fold and resampling of dataset based on stratified sampling (```sklearn.model_selection.StratifiedKFold```) strategy according to the class label to ensure the balance between classes.

Following are the important configurations for training in ```config.gin```:
<pre>
Graham = False # whether to use self-build Graham process

num_classes = 2 # 1 for regression; 2 for binary classification ; 5 for multi(5-)class classification

batch_size = 30 # batch size

Trainer.N_epochs = 1000

Trainer.num_classes=%num_classes

Trainer.lr = 2e-4 # learning rate

Trainer.Ckpts = True # whether to record checkpoint

Trainer.Wandb=True # upload data to the Wandb or not

Trainer.Wandb_Foto= True # upload the recorded pictures to the Wandb or not, which can save the disk capacity if closed

Trainer.Profiler = False 
# whether to record the profiler. Highly recommended to be closed to save RAM especially for training on Kaggle dataset

load_file_names.num_sample = 700
# size of training set after resampling. Each class will be num_sample/num_classes

fold_number=5
# choose how many folds will be set on the raw dataset
# in normal train process it means 20% of the raw dataset will be split out as validation dataset

</pre>

## Evaluate
In this mode the pretrained models can be evaluated by the confusion metrics and deep visualisation. The checkpoints which is already built by the training steps will be restored to the corresponding model. Here you can specify which checkpoint file in the experiments will be evaluated and which picture in the test dataset will be deep visualized as example in ```config.gin``` as follows:
<pre>
ckpt_fn="run_best_EfficientNetB2(G)"
# Specify which checkpoint file in the experiments will be evaluated
</pre>
<pre>
Visualisation.image_id = "IDRiD_054.jpg"
# choose which picture will be used to show the deep visualisation
</pre>
Note that even if the model selected by the  ```main.py``` is different from the model for the checkpoint, it has lower priority than the model name which is found in the ```ckpt_fn``` of the configuration. To use the self-build Graham preprocessed test set to evaluate the model which is trained in this type of train set the ```(G)``` in filename shall also be recognizable.
## Tune
In this Mode you can tune the ViT-16 Model with 5-class classification using ```tensorboard.plugins.hparams```
## Wandb
In this Mode you can tune the ViT-16 Model with 5-class classification using ```wandb.sweep```, which is more advanced than using tensorboard and recommended.
## Ensemble
Here you can type in the checkpoint filenames of the models you've trained to do the ensemble learning. All these checkpoints will be restored to the corresponding model structures and do the ensemble learning (such as voting):
<pre>
ckpt_fns=['run_best_EfficientNetB2' ,'run_best_MobilenetV2','run_best_ResNet50','run_best_DenseNet121','run_best_InceptionV3']
# Specify which checkpoint files will be used in ensemble learning.
</pre>
if the type of parameter is not a list but string, the ensemble learning will be done on a k_fold experiment record like following:
<pre>
ckpt_fns='run_2021.12.22_T_12-33-04ResNet18_10-Fold'
</pre>
## Kaggle
Here you can use the training system to train the models on Kaggle challenge dataset. Since the configurations are much more different from the one used before, another gin file is build for it, namely ```config_Kaggle.gin```

## Results

Results of different models for binary classification:
The best accuracy on test set could be 87.37%(Ensemble model)

|  |Sensitivity/Recall |Specificity |Precision | Accuracy |  Balanced-accuracy |   F1 score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 |78.12%|94.87%|96.15%|84.46%|86.49%|86.21%|
| ResNet34 |79.69%|92.31%|94.44%|84.46%|85.00%|86.44%
|ResNet50|79.69%|97.44%|98.08%|86.40%|88.55%|87.93%
| MobileNetV2 | 76.56%|92.31%|94.23%|82.52%|84.44%|84.48%|
| InceptionV3 |81.25%|82.05%|88.14%|81.54%|81.64%|84.55%|
| InceptionResNetV2 |78.12%|87.18%|90.91%|81.54%|82.64%|84.03%|
|DenseNet169|78.12%|92.31%|94.34%|83.49%|85.21%|85.47%|
| DenseNet121 |84.38%|79.49%|87.10%|82.51%|81.92%|85.71%|
|EfficientNetB1|78.12%|87.18%|90.91%|81.55%|82.65%|84.03%|
|EfficientNetB2|85.94%|84.62%|90.16%|85.54%|85.27%|88.00%|
| ViT-16 |  82.81%|89.74%| 92.98%|85.43%|86.27%|87.60%
| Voting |84.38%|92.31%|94.74%|__87.37%__|88.33%|89.26%

Deep visualization:
![deep_visual](https://github.tik.uni-stuttgart.de/iss/)


Confusion matrix of multi-class classification(resnet18):
![con_mat](https://github.tik.uni-stuttgart.de/iss/)






