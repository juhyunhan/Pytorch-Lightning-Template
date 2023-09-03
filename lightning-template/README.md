# pytorch lightning Template 

## Tutorial for lightning, hydra 
TEST : CIFAR10 Classification

### 📦 Requirements 

```
pytorch
torchvision
wandb
pytorch_lightning
torchmetrics
... extra you need
``` 

+ implement 😃
    + Configuring Experiments (Hydra)
    + Model Module 
    + Data Module
    + Custom Metrics for Accuracy
    + Wandb logging
    + ...

### How to start
```
python scripts/train.py [+experiments=testexp.yaml]
#[ ] : Option for config overriding
```
When you make overriding config (ex. testexp.yaml), you must write below thing top of config
```
# @package _global_
override things ...
```



### 🙏 Guide for me
* RML_BEVSEG (not published yet) - yelin2
* CVT : Cross View Transformers - https://github.com/bradyz/cross_view_transformers
* pytorch - https://pytorch.org/
* pytorch-lightning - https://www.pytorchlightning.ai/

### 👤 License

* MIT License
* Apache License 2.0
* please notice me other things you know ..
