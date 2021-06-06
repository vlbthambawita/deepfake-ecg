# deepfake-ecg

## [Paper](https://doi.org/10.1101/2021.04.27.21256189) | [GitHub](https://github.com/vlbthambawita/deepfake-ecg) | [Pre-generated ECGs (150k)](https://osf.io/6hved/)
---

## [Pulse2Pulse - development repo](https://github.com/vlbthambawita/Pulse2Pulse)
If you want to train the model from scratch, please refere our development repository Pulse2Pulse.

---

Generate unlimited realistic deepfake ECGs using the deep generative model:Pulse2pulse introduced in our full paper here: https://doi.org/10.1101/2021.04.27.21256189 (DeepFake electrocardiograms: the key for open science for artificial intelligence in medicine) 



## Installation

Use the package manager [pip](https://pypi.org/project/deepfake-ecg/) to install deepfake-ecg generator.



```bash
pip install deepfake-ecg
```

## Usage


The generator functions can generate DeepFake ECGs with 8-lead values  [lead names from first coloum to eighth colum: **'I','II','V1','V2','V3','V4','V5','V6'**] for 10s (5000 values per lead). These 8-leads format can be converted to 12-leads format using the following equations. 

```
lead III value = (lead II value) - (lead I value)
lead aVR value = -0.5*(lead I value + lead II value)
lead aVL value = lead I value - 0.5 * lead II value
lead aVF value = lead II value - 0.5 * lead I value

```

### Run on CPU (default setting)

```python
import deepfakeecg

#deepfakeecg.generate("number of ECG to generate", "Path to generate", "start file ids from this number", "device to run") 

deepfakeecg.generate(5, ".", start_id=0, run_device="cpu") # Generate 5 ECGs to the current folder starting from id=0
```

### Run on GPU

```python
import deepfakeecg

#deepfakeecg.generate("number of ECG to generate", "Path to generate", "start file ids from this number", "device to run") 

deepfakeecg.generate(5, ".", start_id=0, run_device="cuda") # Generate 5 ECGs to the current folder starting from id=0
```
### Pre-generated DeepFake ECGs and corresponding MUSE reports are here: https://osf.io/6hved/
    - In this repository, there are two DeepFake datasets:
        1. 150k dataset - Randomly generated 150k DeepFakeECGs
        2. Filtered all normals dataset - Only "Normal" ECGs filtered using the MUSE analysis report

## A real ECG vs a DeepFake ECG (from left to right):


![GitHub Logo](samples/real_vs_fake_left_to_right_v2.png)

## A sample DeepFake ECG:
![GitHub Logo](samples/2879.png)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Citation:
```latex
@article{ecg-pulse2pulse,
	author = {Thambawita, Vajira Lasantha and Isaksen, Jonas L and Hicks, Steven and Ghouse, Jonas and Ahlberg, Gustav and Linneberg, Allan and Grarup, Niels and Ellervik, Christina and Olesen, Morten Salling and Hansen, Torben and Graff, Claus and Holstein-Rathlou, Niels-Henrik and Str{\"u}mke, Inga and Hammer, Hugo L. and Maleckar, Mary M and Halvorsen, P{\aa}l and Riegler, Michael A. and Kanters, J{\o}rgen K.},
	doi = {10.1101/2021.04.27.21256189},
	elocation-id = {2021.04.27.21256189},
	journal = {medRxiv},
	publisher = {Cold Spring Harbor Laboratory Press},
	title = {DeepFake electrocardiograms: the key for open science for artificial intelligence in medicine},
	url = {https://doi.org/10.1101/2021.04.27.21256189},
	year = {2021}
}
	
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## For more details: 
Please contact: vajira@simula.no, michael@simula.no
