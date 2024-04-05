Training the SELDnet23 model, following the instructions specified [here](https://github.com/sharathadavanne/seld-dcase2023), but using the data generated with `SpatialScaper/experiments/data_generation_TAU_{format}.py` (where format is `MIC` or 'FOA') instead of the synthetic recordings from the DCASE2022 webpage, results in a model with the following performance:

## MIC

```
SED metrics: 
* Error rate: 0.68 [0.62, 0.72]
* F-score: 24.7 [20.88, 28.89]                              
DOA metrics: 
* Localization error: 26.9 [23.66 , 29.43]
* Localization Recall: 45.2 [40.13, 50.50] 
```
