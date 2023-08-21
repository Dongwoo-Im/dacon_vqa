## 월간 데이콘 이미지 기반 질의 응답 AI 경진대회
Dacon : https://dacon.io/competitions/official/236118/overview/description

## Environment
```
OS: Windows 10
CUDA: 11.3
DEVICE: RTX 3070 Ti
```


## Directory
```
├── data
│   ├── image
│   │   ├── train
│   │   │   ├── ...
│   │   └── test
│   │       └── ...
│   └── *.csv
├── model_base.pth
└── ...
```


## Pretrained weight (BLIP)
One-click download link: https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth

Also, pretrained checkpoint (129M & BLIP w/ ViT-B) can be downloaded from https://github.com/salesforce/BLIP#pre-trained-checkpoints


## Scripts
```
conda env create -f environment.yaml
conda activate vqa
python train.py
python inference.py --weight exp0/epoch3_acc0.pt
```


## Citation
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>
