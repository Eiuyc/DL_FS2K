# DL_FS2K
Comparision of several CNN based networks on human face attribute classification. 6 methods related to light-weight convolutional neural networks are implemented on [FS2K](https://github.com/DengPingFan/FS2K), and results are presented in table format as below:


1. Accuracy of 6 attributes
<center>

| method | hair | hair color | gender | earring | smile | frontal face |
| - | :-: | :-: | :-: | :-: | :-: | :-: |
|MobileNetv3 Small | 94.83% | 51.72% | 81.26% | 82.12% | 66.82% | 81.16% |
|MobileNetv3 Large | 94.45% | 50.09% | 83.55% | 78.79% | 66.54% | 81.54% |
|SuffleNet | 95.03% | 39.96% | 60.42% | 82.12% | 64.05% | 83.37% |
|EfficientNetv2 | 95.03% | 51.24% | 84.70% | 82.12% | 66.25% | 83.37% |
|MB3 Small+conv+softmax | 95.02% | 40.15% | 81.17% | 82.50% | 64.15% | 83.37% |
|MB3 Small+linear+softmax | 95.03% | 39.96% | 78.01% | 82.12% | 66.25% | 83.36% |
</center>

2. Average accuracy
<center>

| method | average accuracy |
| - | :-: |
|MobileNetv3 Small | 76.32%|
|MobileNetv3 Large | 75.83%|
|SuffleNet | 70.83%|
|EfficientNetv2 | 77.12%|
|MB3 Small+conv+softmax | 74.39%|
|MB3 Small+linear+softmax | 74.12%|
</center>

## Preparation

1. clone the repo with command
```
git clone https://github.com/Eiuyc/DL_FS2K.git
```
2. download FS2K dataset with command
```
git clone https://github.com/DengPingFan/FS2K.git
```

3. move FS2K under DL_FS2K/data/

The structure of the repo should look as follows:
```
DL_FS2K
├─data
│  └─FS2K # FS2K dataset root directory
│      ├─photo
│      └─sketch
├─doc
├─save # saved models
│ ├─model
│ │ └─ENV2.pth
│ ├─MB3_L.ckpt
│ ├─MB3_S.ckpt
│ ├─MB3_S_conv_softmax.ckpt
│ ├─MB3_S_linear_softmax.ckpt
│ └─SF.ckpt
├─effnetv2.py # efficientNet
├─MB3_L_S.ipynb # mobileNetv3 Large and Small
├─MB3_S_conv_softmax.ipynb # mobileNetv3 Small with convolutional layer branches
├─MB3_S_linear_softmax.ipynb # mobileNetv3 Small with linear layer branches
├─SF.ipynb # SuffleNet
└─train_ENV2.py #  trainning entrypoint
```


## training
- ipynb
  1. open the corresponding *.ipynb file
  2. the train func is defined as below:
   
   ```python
   def train(m, # model
          d, # device
          train_dl, # train dataloader
          val_dl, # validation dataloader
          saveDir=Path('save'), # save directory
          resumePath=None, # resume the last training from this checkpoint
          lr=0.001, # learning rate
          e=50, # total epoch
          s=10 # save checkpoint every s epochs
         ):
   ```
  3. change the `saveDir` to the actual save directory when you execute
  4. execute the code blocks in order
  5. the `best.ckpt` file will be written under `saveDir` after the training

Note the MB3 Small and Large are inplemented on the single file `MB3_L_S.ipynb`, and `mode` is set to `small` as default. Change the `mode=large` in the script when you train MB3 Large:

```python
# for MB3 Small:
class MobileNetV3(nn.Module):
    def __init__(self, nclass=1000, mode='small', width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d):
```
  

- py
  
  EfficientNetv2 can be trained with command:

  ```shell
  python train_ENV2.py
  ```
  the `.pth` file will be written under `./save/model` as default


## validation
- ipynb
  
  execute the val function as below to validate the `.ckpt` file
  ```python
  em, epo, acc = load('save/best.ckpt')
  val(Model().to(d).eval(), em.state_dict(), val_dl, d)
  ```
  
- py
  
  model will be validated automatically while training
