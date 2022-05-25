# Prediction Telco Churn
by Deangelo Bowen 
24 May 2022

---
## Overview
- To observe key drivers of churn at Telco to determine primary causes to why customers are churning.
- Construct a Machine Learning model using classification techniques to predict customer churn for a group of customers.
- Deliver a report that is clear and highlights key findings and insights, while summarizing the processes taken to reach them.
- To answer all questions regarding code, process, key takeaways, and models any stakeholder may have.
    
---

## Project Description
   TelcoCo, a telecommunications company, is rapidly churning customers at a rate of nearly 27% percent. Why are the customers churning? How can they reduce customer churn? The minimum viable product (MVP) is a machine learning model that can accurately predict customer churn and make predictions that can be used to mak recommendations, that can produce potential change which will shape TelcoCo's future.
    
---

## Initial Hypothesis/Questions

- Is there unexpected services related to churn, and why?
- Are those who use mailed checks for payments more or less likely to churn?
- Do customers who have dependents more likely to have longer tenure than those who do not?
- On average when do month to month customers churn? Is there a significant pattern of churns around a certain amount of tenure?
  
---

## Data Dictionary
|Column | Description | Dtype|
    |--------- | --------- | ----------- |
    churn | the target variable: has churned or not | uint8 |
    customer_id | unique id number to identify customer | int64 |
    is_senior_citizen | whether customer is senior citizen | int64 |
    tenure | months with the company | int 64 |
    multiple_lines | whether customer has more than one line | int 64 |
    monthly_charges | customer monthly rate | float64 |
    total_charges | how much customer has paid | float 64 |
    contract_type | one year, two year, month-to-month | object |
    internet_service_type | DSL, Fiber optic, None | object |
    payment_type | how the customer pays balance | object |
    is_male | whether customer is male or not | uint8 |
    has_partner | whether customer has partner or not | uint8 |
    has_dependents | whether customer has dependents or not | uint8 |
    online_security | whether customer has online security or not | uint8 |
    online_backup | whether customer has online back up or not |uint8 |
    phone_service | whether customer has phone service or not | uint8 |
    device_protection | whether customer has device protection or not | uint8 |
    tech_support | whether customer has tech support or not | uint8 |
    streaming_tv | whether customer streams tv or not | uint8  |
    streaming_movies | whether customer streams movies or not | uint |
    paperless_billing | whether customer has paperless billing or not | uint8 |
    one_line | whether customer has one line or not | uint8 |
    no_phone_service | whether customer has no phone service or not | uint8 |
    has_multiple_lines | whether customer has multiple lines or not | uint8 |
    month_to_month_contract | whether customer has month to month or not | uint8 |
    one_year_contract | whether customer has one year contract or not | uint8 |
    two_year_contact | whether customer has two year contract or not | uint8 |
    dsl_internet | whether customer has dsl internet or not | uint8 |
    fiber_optic_internet | whether custome rhas fiber optic or not | uint8 |
    no_internet_service | whether a customer has no internet or not | uint8 |
    bank_transfer_autopay | whether a customer pays via bank transfer | uint8 |
    credit_card_autopay | whether a customer pays via credit card | uint8 |
    electronic_check_nonauto | whether a customer pays via e-check | uint8 |
    mailed_check_nonauto | whether a customer pays via mailed check | uint8 |
    is_autopay | whether a customer has autopay or not | uint8 |
    
---
## Project Plan

   Recreate the plan by following these steps
   
### Planning
    - Define goals
    - Determine audience and delivery format
    - Ask questions/formulate hypothesis
    - Determine the MVP

### Acquisition
    - Create a function that establishes connection to telco_churn_db
    - Create a function that holds your SQL query and reads results
    - Creating functing for caching data and stores as .csv for ease
    - Create and save in acquire.py so functions can be imported

   ### Preparation
    - Identified each of the features data types and manipulated relevant columns to      integers.
    - Removed all irrelevant or duplicated columns.
    - Encoded to numerical values (Changed Yes's to 1, No's to 0)
    - Renamed columns to more appropriate and identifiable naming conventions.
    - Repeated these steps on the split data for future modeling.
    
   ### Exploration
    - Use the initial questions to guide the exploration process
    - Create visualizations to help identify drivers
    - Use statistical testing to confirm or deny hypothesis
    - Document answers to questions as takewaways
    - Utilize explore.py as needed for clean final report

 ### Model
    - Train model
    - Make predictions
    - Evaluate model
    - Compute accuracy
    - Utilize model.py as needed for clean final report
--- 

### Key Findings and takeaways
    - Fiber Optic Internet has a major correlation with churn exceeding DSL and No internet service customers combined.
    - There is a payment type strongly related to churn. Customers who pay by electronic checks are churning faster than any other category.
    - Customers with dependents have on average stayed with Telco longer than those without.
    - An overwhelming majority of month-to-month contract customers are churning before they've reached 10 months tenure.
    - Logistic Regression model produced to operate over baseline accuracy by 8%. 

### To Recreation of Project:
    - You will need an env file with database credentials saved to your working directory
        - database credentials with CodeUp database (username, password, hostname) 
    - Create a gitignore with env file inside to prevent sharing of credentials
    - Download the acquire.py and prepare.py (model.py and explore.py are optional) files to working directory
    - Create a final notebook to your working directory
    - Review this README.md
    - Libraries used are pandas, matplotlib, Scipy, sklearn, seaborn, and numpy
    - Run telco_final.ipynb
    - Create a .csv file with the predictions from your ML model of at risk customers
