# LoadCNN
### Purpose
- experiment with various input sequence lengths and output sequence lengths to figure out the performance changes according to sequence characteristics.

### Experiments
- Model performance comparison
- Performance comparison according to the sequence length
- Performance comparison according to the target time slo

### Dataset
- CER Smart Meter Project
  - Large-scale smart meter dataset from Smart Metering Electricity Customer Behaviour Trials (CBTs).
  - This dataset was collected from more than 5,000 Irish customers.
  - Electricity consumption (kWh) data sampled every 30 minutes for 536 days form July 1, 2009.
  - Selected 929 most representative customers.
  - Unobserved values were treated as missing values.
  - The test set contains the last 30 days and the validation set contains 60 days except for test data from the last 90 days.
  - The training set contains all of the rest data.
  - Electricity consumption data was scaled by dividing by 10.
<img width="800" alt="image" src="https://user-images.githubusercontent.com/62350977/143507091-1f3939d7-7f81-47f7-a025-5dcfcb6f0d58.png">

### Experiment setting
<img width="800" alt="image" src="https://user-images.githubusercontent.com/62350977/143507114-84895517-e8bb-4474-9fea-e2d83e6eb225.png">

### Run

```Python
python main_simple.py --save_root save_root --days 7
```


### Metrics
- MSE
- RMSE
- MAE
- NRMSE

### Experimental results
<img width="1000" alt="image" src="https://user-images.githubusercontent.com/62350977/143507155-c1bb7420-5680-43cc-9dda-003269c05c37.png">
<img width="1000" alt="image" src="https://user-images.githubusercontent.com/62350977/143507173-bed20b1d-e936-4b68-890f-909a5669b8dc.png">
<img width="1000" alt="image" src="https://user-images.githubusercontent.com/62350977/143507193-21aac3db-72d0-4fa5-a429-f455ff57b938.png">

### Reference
-  LoadCNN: A Efficient Green Deep Learning Model for Day-ahead Individual Resident Load Forecasting.



