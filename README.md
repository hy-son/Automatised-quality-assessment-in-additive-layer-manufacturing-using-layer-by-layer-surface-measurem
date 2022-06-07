# Automatised quality assessment in additive layer manufacturing using layer-by-layer surface measurements and deep learning
___
[CIRP ICME 2020 paper: Automatised quality assessment in additive layer manufacturing using layer-by-layer surface measurements and deep learning](https://doi.org/10.1016/j.procir.2021.03.050)
Additive manufacturing (AM) has gained high research interests in the past but comes with some drawbacks, such as the difficulty to do in-situ quality monitoring. In this paper, deep learning is used on electron-optical images taken during the Electron Beam Melting (EBM) process to classify the quality of AM layers to achieve automatized quality assessment. A comparative study of several mainstream Convolutional Neural Networks to classify the images has been conducted. The classification accuracy is up to 95 %, which demonstrates the great potential to support in-process layer quality control of EBM.And the error analysis has shown that some human misclassification were correctly classified by the Convolutional Neural Networks.

## Getting started
___
### Installation
- Clone this repo:
`git clone https://github.com/hy-son/Images-classification-extraction`
- Create a new conda environement (Python 3.6)
`conda env create -f environment.yml`

Some import may fail, to fix them please run:
- `pip install --force-reinstall --no-cache-dir numpy`
- `pip install --force-reinstall --no-cache-dir --user torchvision`
  
### Configuration:
All configuration data are stored in `data.py`.
**model_name and trainned_dir must be list of the same size** as for the n model_name the n trainned_dir file will be loaded.

### Data
The ELO images data must be stored in the `Data` folder, split in 3 folders `train`, `test`, `val` and all images must be in a folder with of there class.
Ex: `Data\train\0\10-44-52_01.jpg`
The required images size is 224 by 224.
You can download a small data sample here: https://drive.google.com/file/d/1yoarg6nhrNrUypNiRZ8vsZ9uN2Ji6gsQ/view?usp=sharing

### Use
You can test the accuracy of the neural network by running `main.py`.
The jupyter notebook `Classification extraction execution time.ipynb` will allow you to create the confusion matrix.
**To train** launch 5CNN.py, (the trainning parameters are stored in his dataclass Project_data).

### Training
The 5 networks can be trained as show in `5CNN.codeexaple.py`

## Citation
If you find this code useful, please consider citing my paper.
[DOI](https://doi.org/10.1016/j.procir.2021.03.050)
```
 @article{le roux_liu_ji_kerfriden_gage_feyer_körner_bigot_2021,
  title={Automatised quality assessment in additive layer manufacturing using layer-by-layer surface measurements and deep learning},
  volume={99},
  DOI={10.1016/j.procir.2021.03.050},
  journal={Procedia CIRP}, 
  author={Le Roux, Léopold and Liu, Chao and Ji, Ze and Kerfriden, Pierre and Gage, Daniel and Feyer, Felix and Körner, Carolin and Bigot, Samuel},
  year={2021}, 
  pages={342–347}} 
```
