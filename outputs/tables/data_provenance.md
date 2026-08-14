# Data Provenance

- **source:** C:\Sriram Churn\IBM_Telco_customer_churn_IBM_dataset.csv
- **sha256:** 700af502733c9ebde7678e120f02eec0a99de50ecb8abbd97e281e309583c08c
- **unit:** one row per telco customer
- **raw_rows:** 7043
- **prepared_rows:** 7043
- **final_shape:** [7043, 29]
- **target_column:** Churn Value
- **id_column:** CustomerID
- **feature_count:** 27
- **dropped_columns:** ['Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason']
- **cleaning_actions:** ['removed rows with missing target only', 'coerced Total Charges to numeric', 'created derived features', 'retained missing inputs for pipeline imputation']
- **dataset_note:** The project uses one consolidated IBM Telco CSV because the original multi-file source could not be obtained reliably.
- **limitation:** The Iranian churn file is schema-incompatible and was not used for direct validation.