# deepfake-ecg
Generate unlimited realistic deepfake ECGs.  

## Installation

Use the package manager [pip](https://pypi.org/project/deepfake-ecg/) to install deepfake-ecg.



```bash
pip install deepfake-ecg
```

## Usage


The following fucntions can generate Deep Fake ECGs with 8-lead values  [lead names from first coloum to eighth colum: **'I','II','V1','V2','V3','V4','V5','V6'**] for 10s (5000 values per lead). These 8-leads format can be converted to 12-leads format using the following equations. 

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


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## For more details: 
Please contact: vajira@simula.no, michael@simula.no

## License
[MIT](https://choosealicense.com/licenses/mit/)
