# deepfake-ecg
Generate unlimited realistic deepfake ECGs.  

## Installation

Use the package manager [pip](https://pypi.org/project/deepfake-ecg/) to install deepfake-ecg.



```bash
pip install deepfake-ecg
```

## Usage

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
