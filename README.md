There are two files:

* **train.py** - to train a model on train dataset. To run it use 
 
              python train.py -train_datapath hymenoptera_data/train -model "VGG" -pretrained True -aug True

-train_datapath - path to your train data

-model - choose type of model: "VGG" or "resnet18"

-pretrained - choose whether to use fine-tuning or not: "True" or "False"

-aug - choose whether to use data augmentation (only for models with fine-tuning): "True" or "False"

Checkpoints will be saved to folder 'checkpoints/'.

***

* **eval.py** - to get a model quality on test data. To run it use 
 
              python eval.py -test_datapath hymenoptera_data/val -model "VGG" -pretrained True 

-test_datapath - path to your test data

-model - choose type of model: "VGG" or "resnet18"

-pretrained - choose whether to use fine-tuning or not: "True" or "False"


