# [Machine Learning] Churn Prediction & Segmentation

# I. Introduction
This project uses two machine learning approaches for different purposes: 
- Supervised learning to predict which customers are likely to churn
- Unsupervised learning to segment customers who have churned

# II. Dataset
Labeled data of an eCommerce store, each row represents a single customer and their information:
- CustomerID: Unique customer ID
- Churn: Churn flag (1:Churned; 0:No churn)
- Tenure: Tenure of customer in organization
- PreferredLoginDevice: Preferred login device of customer
- CityTier: Region code
- WarehouseToHome: Distance in between warehouse to home of customer
- PreferPayment: The payment method that user usually use
- Gender: Gender of customer
- HourSpendOnApp: Number of hours spend on mobile application or website
- NumberOfDeviceRegistered: Total number of devices is registered on particular customer
- PreferedOrderCat: Preferred order category of customer in last month
- SatisfactionScore: Satisfactory score of customer on service
- MaritalStatus: Marital status of customer
- NumberOfAddress: Total number of added added on particular customer
- Complain: Any complaint has been raised in last month (1:Yes; 0:No)
- OrderAmountHikeFromlastYear: Order amount hike from last year
- CouponUsed: Number of coupons customer has used
- OrderCount: Number of orders
- DaySinceLastOrder: Number of days since last order
- CashbackAmount: Total cashback amount

# III. Exploring the dataset
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/00_data_process.ipynb)
- First look:
<img width="661" height="507" alt="image" src="https://github.com/user-attachments/assets/b1c9f8a0-54e0-4288-87c6-cea462784458" />

- Numeric columns summary:
<img width="667" height="672" alt="image" src="https://github.com/user-attachments/assets/71efdd16-fb0c-4d31-b3b3-0c384bfad93b" />

- Non-numeric columns summary:
<img width="755" height="312" alt="image" src="https://github.com/user-attachments/assets/d7a09ced-6a3f-434c-a4d1-b839ed8aa45f" />

- NA values map:
<img width="1292" height="592" alt="image" src="https://github.com/user-attachments/assets/71aebe7a-8fd8-4313-aef1-1d0b2ab1d64c" />

- With non-numeric columns, I replace values with similar meaning into one single value only (ex: "CC" and "Credit Card" -> "CC")
<img width="587" height="317" alt="image" src="https://github.com/user-attachments/assets/4ba21f16-c32d-432b-bda5-9c168015946e" />

- With numeric columns, handling missing values will be executed right before fitting ML models, filling missing values now would make visualization less accurate
# IV. Visualization
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/01_eda.ipynb)
- First, I used the one-hot encoding method to transform non-numeric features:
```python
df = pd.get_dummies(df, drop_first = True)
```
- In order to determine which features are important, I fit 70% of the dataset with the Random Forest model and extract features with importance score above 0.05
  + Model performance on test set:
<img width="505" height="190" alt="image" src="https://github.com/user-attachments/assets/fe9bba56-233b-42a3-adf8-53bc6fb8fb3f" />

  + Importance score:
<img width="816" height="432" alt="image" src="https://github.com/user-attachments/assets/f87684c7-5345-48a0-85dc-ae30c351b10d" />

- Compare distribution within 2 groups: Churn and No churn with important features above
<img width="1511" height="811" alt="image" src="https://github.com/user-attachments/assets/5db8e540-44ab-4b2c-8109-8fe5962303a2" />
<img width="567" height="433" alt="image" src="https://github.com/user-attachments/assets/1a37eed7-2316-411f-a786-dc454659a6fa" />

- Churn rate of different groups:
<img width="1610" height="1011" alt="image" src="https://github.com/user-attachments/assets/ce8d1787-3fca-4ada-95c6-cde3f9554102" />

- ***Key takaways:***
  - Insights:
    + New users tend to churn more
    + Users with lower Cashback Amount are more likely to churn more
    + Customers with high number of addresses & long time no order have higher chance of churn
  - Suggestions:
    + Focus on new customers and customers with last order > 1 month, launch cashback promotions for them to acquire/retain
    + Aware of customers with high number of addresses, limit the number of addresses can be added or implement verify identity in order to reduce the risk of fraud

# V. Building prediction model
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/02_fit_model.ipynb)
- In this section, I split the dataset into 3 parts: train/valid/test with 70/10/20 ratio
- The dataset is quite imbalanced, so I adjust the class weight hyperparameters for better performance and avoid bias
<img width="305" height="95" alt="image" src="https://github.com/user-attachments/assets/0a198249-832f-46b2-8fbb-a9c8c7fb7cac" />

```python
churn_wt = y_train.count()/y_train.sum() - 1 # y_train contains Target feature (Churn) only
all_mod = {"Random Forest"  : RandomForestClassifier(random_state = 10, class_weight = {0:1, 1:churn_wt}),
           "Hist Gradient Boost" : HistGradientBoostingClassifier(random_state = 10, class_weight = {0:1, 1:churn_wt}),
           "XGBoost" : XGBClassifier(random_state = 10, scale_pos_weight = churn_wt)
            }
```
- After fitting 3 models on the training set, here are performance metrics on the valid set:
<img width="716" height="434" alt="image" src="https://github.com/user-attachments/assets/39150cf7-b8ca-4079-8d8c-83a8ffaee9fb" />

- The XGBoost model performs extremely well, so I tune this model using the Optuna library and here are the result on the test set after tuning:
<img width="507" height="187" alt="image" src="https://github.com/user-attachments/assets/bb26c05c-02f1-4977-b9d9-f3fd96c9df38" />

- ROC curves:
<img width="459" height="469" alt="image" src="https://github.com/user-attachments/assets/b116d6e6-2d2c-42e4-b3d3-79481886f410" />

- SHAP values:
<img width="914" height="498" alt="image" src="https://github.com/user-attachments/assets/f99a332e-c1da-42c7-aaa2-794c97d4b8e2" />

- ***Key takaways:***
  + As usual, XGBoost work extremely well on tabular dataset
  + Base on SHAP values, we can easily explain machine learning predictions by measuring how much each input feature pushes the model's output up or down

# VI. Segmentation
- [See full notebook](https://github.com/DTN-492/ML_Churn_Predict_and_Segmentation/blob/main/notebooks/03_clustering.ipynb)
- In this section, I use K-means model to separate Churned customer
- To reduce the dimension, I use PCA with 9 components (explain ~ 80% of variance)
<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/6e5d0696-304e-4437-9f10-49f9a2cb5e29" />

- Since K-means model sensitive to outliers, scaling is an important preprocessing step, and here is the WCSS chart after fitting the K-means model on the scaled PCA dataset:
<img width="860" height="469" alt="image" src="https://github.com/user-attachments/assets/d0837994-b45f-4b9f-8cea-27211c96ad7f" />
<img width="851" height="469" alt="image" src="https://github.com/user-attachments/assets/d7c0d527-602e-4e85-978c-1dd3f42f6377" />

-> There is no optimal value for k, so segmentation churned customers is not necessary.

# VII. Conclusion
- Machine learning is valuable for a range of tasks, including prediction, customer segmentation, and identifying important features.
- XGBoost is a powerful model for tabular data, but effective data cleaning and feature engineering are essential for strong performance.
- SHAP values help address the opacity of black-box models by making their predictions more interpretable.
- Customer segmentation is not always necessary: churned customers may share similar behaviors, making them easier to identify.
