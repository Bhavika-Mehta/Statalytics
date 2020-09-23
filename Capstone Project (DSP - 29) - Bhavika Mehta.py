#!/usr/bin/env python
# coding: utf-8

# # Calling Libraries

# In[1]:


import pandas as pd
import numpy as np
from numpy import percentile
from numpy.random import seed
from numpy.random import randn
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import scipy
from scipy import stats
from scipy.stats import t


# # Importing Data

# In[2]:


data = pd.read_table("/Users/bhavikamehta/Downloads/Python Project/data.txt")


# # Checking Summary

# In[3]:


pd.pandas.set_option('display.max_columns',None)
data.describe(include = "all")


# # Checking how data looks like

# In[4]:


data.head(10)


# In[5]:


data.shape


# In[6]:


# Checking data type of each variable
pd.pandas.set_option('display.max_columns',None)
data.dtypes.head(60)


# # Data Encoding

# In[7]:


# Removing months element from the observations
data["term"] = data["term"].replace({'months':''}, regex = True)
# Removing and years and spaces from the observations
data["emp_length"] = data["emp_length"].replace({'years':'','year':'',' ':'','<':'','\+':''}, regex = True)
# Maintaing uniformity in vertification status
data["verification_status"] = data["verification_status"].replace({"Verified": 1,"Source Verified": 1,"Not Verified": 0})


# # Changing necessary data types to Date and Time

# In[8]:


data["issue_d"] = pd.to_datetime(data["issue_d"])
data["earliest_cr_line"] = pd.to_datetime(data["earliest_cr_line"])
data["last_pymnt_d"] = pd.to_datetime(data["last_pymnt_d"])
data["next_pymnt_d"] = pd.to_datetime(data["next_pymnt_d"])
data["last_credit_pull_d"] = pd.to_datetime(data["last_credit_pull_d"])


# # Finding correlation between Numerical Variables

# In[9]:


corrmatrix = data.corr()
plt.figure(figsize=(50,50))
sns.set_context("paper", font_scale=2)
sns.heatmap(corrmatrix,annot = True,cmap='bwr',vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.show()


# In[10]:


corrmatrix


# Removing the columns that have high correlation and deduce almost the same information

# In[11]:


data = data.drop(["id", "funded_amnt","funded_amnt_inv", "out_prncp_inv", "total_pymnt_inv"], axis = 1)


# # Visualizing Missing Values in the Data with Heatmap

# In[12]:


sns.heatmap(data.isnull(), cbar = False, cmap = "viridis")


# # Checking the number of missing values and it's percentage in each column

# In[13]:


missing_data = pd.DataFrame({"Count" : data.isna().sum(), "Percent" :  100*(data.isna().sum()/ len(data))})
missing_data = missing_data[missing_data["Count"] > 0]
missing_data.sort_values(by = ["Count", "Percent"], ascending = False)


# Checking the count of joint and individual application type

# In[14]:


data["application_type"].value_counts()


# Since there are very few observations for joint application type, we choose to drop columns relating joint field as they also contain more than 99% missing data

# In[15]:


data = data.drop(["dti_joint", "annual_inc_joint", "verification_status_joint", "application_type"], axis = 1)


# Dropping columns that do not reveal much information due to huge missing value percentage

# In[16]:


data = data.drop(["il_util", "mths_since_rcnt_il", "open_acc_6m", "open_il_6m", "open_il_12m","open_il_24m","total_bal_il","open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m", "desc", "mths_since_last_record" ], axis = 1)


# Dropping variables that don't seem to be of much value

# In[17]:


data = data.drop(["member_id", "zip_code", "addr_state", "sub_grade", "title"], axis = 1)


# # Imputation of Missing Values

# Imputing "Month since last major derog"

# In[18]:


plt.figure(figsize= (14,9))
sns.boxplot(x = "grade", y = "mths_since_last_major_derog", data = data, palette = "winter", order = ["A", "B", "C", "D", "E", "F", "G"])


# In[19]:


# Checking whether there is a significant relationship between "grade" and "mths_since_last_delinq" by applying annova test
formula = 'mths_since_last_major_derog ~ C(grade)'
model = ols(formula, data).fit()
aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
print(aov_table)


# In[20]:


#Finding the mean and median of "Month since last major derogacy" wrt grade
data.groupby("grade")["mths_since_last_major_derog"].agg(["mean", "median"])


# In[21]:


# Defining function for imputation for the column of "mths_since_last_major_derog" by mean of grade's group
def impute_last_derog(cols):
    mths_since_last_major_derog = cols[0]
    grade = cols[1]
    
    if pd.isnull(mths_since_last_major_derog):
        if grade == "A":
            return 44.399147
        elif grade == "B":
            return 43.818358
        elif grade == "C":
            return 44.338775
        elif grade == "D":
            return 44.102446
        elif grade == "E":
            return 44.071474
        elif grade == "F":
            return 43.644042
        else:
            return 42.878369
        
    else:
        return mths_since_last_major_derog
        


# In[22]:


# Imputing the column of "mths_since_last_major_derog" by mean of grade's group
data["mths_since_last_major_derog"] = data[["mths_since_last_major_derog", "grade"]].apply(impute_last_derog, axis = 1)


# Imputing "Month since last delinq" wrt Grade

# In[23]:


sns.barplot(x="grade", y="mths_since_last_delinq" ,data = data, color = 'salmon' ,order= ["A", "B", "C", "D", "E", "F", "G"])


# In[24]:


plt.figure(figsize= (12,7))
sns.boxplot(x = "grade", y = "mths_since_last_delinq", data = data, palette = "winter")


# In[25]:


# Checking whether there is a significant relationship between "grade" and "mths_since_last_delinq" by applying annova test
formula = 'mths_since_last_delinq ~ C(grade)'
model = ols(formula, data).fit()
aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
print(aov_table)


# In[26]:


# Finding the mean and median of month since last delinq wrt grade
data.groupby("grade")["mths_since_last_delinq"].agg(["mean", "median"])


# In[27]:


# Defining function for imputation for the column of "mths_since_last_major_derog" by mean of grade's group
def impute_last_delinq(cols):
    mths_since_last_delinq = cols[0]
    grade = cols[1]
    
    if pd.isnull(mths_since_last_delinq):
        if grade == "A":
            return 34
        elif grade == "B":
            return 31
        elif grade == "C":
            return 30
        elif grade == "D":
            return 30
        elif grade == "E":
            return 30
        elif grade == "F":
            return 29
        else:
            return 27
        
    else:
        return mths_since_last_delinq
        


# In[28]:


# Imputing the column of "mths_since_last_major_derog" by mean of grade's group
data["mths_since_last_delinq"] = data[["mths_since_last_delinq", "grade"]].apply(impute_last_delinq, axis = 1)


# Imputation of "Total Curent Balance"

# In[29]:


sns.distplot(data["tot_cur_bal"])


# In[30]:


data["tot_cur_bal"].agg(["mean", "min", "max", "median"])


# In[31]:


data["tot_cur_bal"] = data["tot_cur_bal"].fillna(data["tot_cur_bal"].median())


# Imputation of "Total Collection Amount"

# In[32]:


data['tot_coll_amt'].hist(bins=50,color = 'salmon' )


# In[33]:


data["tot_coll_amt"].agg(["min", "max", "mean", "median"])


# In[34]:


data["tot_coll_amt"] = data["tot_coll_amt"].fillna(data["tot_coll_amt"].median())


# Imputing "Total revolving high limit"

# In[35]:


sns.distplot(data["total_rev_hi_lim"], color = "green")


# In[36]:


# Checking the relationship between "Total_rev_hi_lim" wrt default 
formula = 'total_rev_hi_lim ~ C(default_ind)'
model = ols(formula, data).fit()
aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
print(aov_table)


# In[37]:


data.groupby("default_ind")["total_rev_hi_lim"].agg(["mean", "median"])


# In[38]:


data["total_rev_hi_lim"] = np.where(data["default_ind"] == 1, 20500.0, 24000.0)


# Dropping "emp_title" as it does not reveal too much information and has many different categories

# In[39]:


data = data.drop(["emp_title"], axis = 1)


# Imputing "Employment Length"

# In[40]:


data["emp_length"] = pd.to_numeric(data["emp_length"])


# In[41]:


data['emp_length'].value_counts().plot.bar(color = "blue")


# In[42]:


plt.figure(figsize= (12,7))
sns.boxplot(x = "emp_length", y = "loan_amnt", data = data, palette = "winter")


# In[43]:


data.groupby("emp_length")["loan_amnt"].agg(["mean", "median"])


# In[44]:


# Defining function for imputation for the column of "emp_length" by median of loan amount group
def impute_emp_length(cols):
    emp_length = cols[0]
    loan_amnt = cols[1]
    
    if pd.isnull(emp_length):
        if loan_amnt <= 12000.0:
            return 3.0
        elif (loan_amnt > 12000.0) & (loan_amnt <= 12600.0):
            return 6.0
        elif (loan_amnt > 12600.0) & (loan_amnt <= 13200.0):
            return 7.0
        elif (loan_amnt > 13200.0) & (loan_amnt <= 13600.0):
            return 8.0
        elif (loan_amnt > 13600.0) & (loan_amnt <= 14000.0):
            return 9.0
        
        else:
            return 10.0
        
    else:
        return emp_length
        


# In[45]:


data["emp_length"] = data[["emp_length", "loan_amnt"]].apply(impute_emp_length, axis = 1)


# Imputing "Revolving Utilization"

# In[46]:


sns.distplot(data["revol_util"], color = "orange")


# In[47]:


data["revol_util"].agg(["mean", "median", "min", "max"])


# In[48]:


data["revol_util"] = data["revol_util"].fillna(data["revol_util"].median())


# Dropping missing values in "Collection 12 month excluding medical"

# In[49]:


data["collections_12_mths_ex_med"].fillna(0, inplace = True)


# Checking outliers per column

# In[50]:


data1 = data._get_numeric_data()
q1 = data1.quantile(0.25)
q3 = data1.quantile(0.75)
iqr = q3 - q1
((data1<(q1 - 1.5*iqr))| (data1 > (q3 +1.5* iqr))).sum().sort_values()


# # Exploratory Data Analysis

# Graph between "Default_ind" and "Interest Rate"

# In[51]:


plt.figure(figsize= (12,7))
sns.boxplot(x = "default_ind", y = "int_rate", data = data, palette = "winter")

# Higher interest rate led to more default cases


# Checking the distribution of "Loan Amount"

# In[52]:


sns.distplot(data.loc[data['loan_amnt'].notnull(), 'loan_amnt'], kde=True)


# Graph between "Default_ind" and "Loan Amount"

# In[53]:


data["default_ind"] = data["default_ind"].astype('category')
sns.boxplot(x='loan_amnt', y='default_ind', data=data)

# Non-Defaulters had higher annual income


# Graph for "Interest Rate"

# In[54]:


sns.distplot(data.loc[data['int_rate'].notnull(), 'int_rate'], kde=True)


# Graph for "Installment"

# In[55]:


sns.distplot(data.loc[data['installment'].notnull(), 'installment'], kde=True, color = "red")


# Graph between "Installment" and "Default_ind"

# In[56]:


data.groupby('default_ind')['installment'].describe()


# In[57]:


sns.boxplot(x='installment', y='default_ind', data=data)

# Higher installments led to more default cases


# CountPlot showcasing the distribution of "Grade"

# In[58]:


sns.countplot(data['grade'], order=sorted(data['grade'].unique()), palette = "winter")


# Bar Plot between "Default_ind" and "Grade"

# In[59]:


plt.figure(figsize=(50,50))
charge_off_rates = data.groupby(['grade'])['default_ind'].value_counts(normalize=True)
sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, saturation=1)

# As the grade inceases from A to G, the number of default cases also increase.


# CountPlot showcasing the distribution of "Purpose"

# In[60]:


sns.countplot(data['purpose'], order=sorted(data['purpose'].unique()), saturation=1)


# In[62]:


data.groupby('purpose')['default_ind'].value_counts(normalize=True)


# Plot for distribution of "dti"

# In[63]:


sns.distplot(data.loc[data['dti'].notnull(), 'dti'], kde=False)


# In[64]:


data.groupby('default_ind')['dti'].describe()


# In[65]:


sns.boxplot(x='dti', y='default_ind', data=data)

# Dti ratio for for Defaulters is higher.


# Distribution of "Dti"

# In[68]:


plt.figure(figsize=(8,3), dpi=90)
sns.distplot(data.loc[data['dti'].notnull() & (data['dti']<60), 'dti'], bins = 30, kde = False)
plt.xlabel('Debt-to-income Ratio')
plt.ylabel('Count')
plt.title('Debt-to-income Ratio')


# Distribution for "Earliest Credit Line"

# In[70]:


sns.distplot(data.loc[data['earliest_cr_line'].notnull(), 'earliest_cr_line'], kde=True, color = "red", bins = 30)


# Distribution of "Open Account" 

# In[71]:


plt.figure(figsize=(10,3), dpi=90)
sns.countplot(data['open_acc'], order=sorted(data['open_acc'].unique()), saturation=1)
_, _ = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))
plt.title('Number of Open Credit Lines')


# In[72]:


data.groupby('default_ind')['open_acc'].describe()

# Number of open credit lines in case of defaulters was higher.


# Description and distribution for "Revolving Balance"

# In[73]:


data['revol_bal'].describe()


# In[74]:


# Log Transformation

data['log_revol_bal'] = data['revol_bal'].apply(lambda x: np.log10(x+1))


# In[75]:


data.drop('revol_bal', axis=1, inplace=True)


# In[76]:


sns.distplot(data.loc[data['log_revol_bal'].notnull(), 'log_revol_bal'], kde=False, bins = 30)


# In[77]:


sns.boxplot(x='log_revol_bal', y='default_ind', data=data)

# Revolving Balance for defaulters is higher.


# In[78]:


data.groupby('default_ind')['log_revol_bal'].describe()


# Graph for "Revolving Utilization"

# In[79]:


data['revol_util'].describe()


# In[80]:


sns.distplot(data.loc[data['revol_util'].notnull(), 'revol_util'], kde=True)


# In[81]:


sns.boxplot(x='revol_util', y='default_ind', data=data)

# Revolving Utilization is higher in the case of Defaulters.


# Distribution of "Total Account"

# In[82]:


plt.figure(figsize=(12,3), dpi=90)
sns.countplot(data['total_acc'], order=sorted(data['total_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 176, 10), np.arange(0, 176, 10))
plt.title('Total Number of Credit Lines')


# In[83]:


data.groupby('default_ind')['total_acc'].describe()

# Total number of accounts is higher for Defaulters.


# Graph between "Default_ind" and "Initial List Status"

# In[84]:


sns.set_style("whitegrid")
sns.countplot(x = "initial_list_status", hue = "default_ind", data = data, palette = "winter")

# Cases of default are more where the initial list status is "f"


# Relationship between "Outstanding Principal" and "Default_ind"

# In[85]:


data['out_prncp'].describe()


# In[86]:


data.groupby('default_ind')['out_prncp'].describe()


# Relationship between "Total Payment" and "Default_ind"

# In[87]:


data.groupby('default_ind')['total_pymnt'].describe()


# Relationship between "Total received interest" and "Default_ind"

# In[88]:


data.groupby('default_ind')['total_rec_int'].describe()

# Interest received in case of Defaulters was much higher than that for Non-Defaulters


# Relationship between "Total received late fee" and "Default_ind"

# In[89]:


data.groupby('default_ind')['total_rec_late_fee'].describe()

# Total late fee received was much higher for Defaluters.


# Relationship between "Recoveries" and "Default_ind"

# In[90]:


data.groupby('default_ind')['recoveries'].describe()

# Recoveries for Defaulters was much higher than in the case of Non-Defaulters


# Relationship between "Account now Delinquent" and "Default_ind"

# In[91]:


data.groupby('default_ind')['acc_now_delinq'].describe()


# Relationship between "Total collection amount" and "Default_ind"

# In[92]:


data.groupby('default_ind')['tot_coll_amt'].describe() 

# Mean collection amount is less for Non-Defaulters


# Graph between "Term" and "Default_ind"

# In[93]:


sns.set_style("whitegrid")
sns.countplot(x = "term", hue = "default_ind", data = data, palette = "RdBu_r")


# Distribution for "Total Payment"

# In[94]:


sns.distplot(data["total_pymnt"], kde = True, color = "red", bins = 30)


# In[95]:


data.columns


# Data encoding for "Grade" and "Payment Plan"

# In[96]:


data["grade"] = data["grade"].replace({"A": 1,"B": 2,"C": 3, "D":4, "E":5, "F":6, "G" :7})
data["pymnt_plan"] = data["pymnt_plan"].replace({"y" : 1, "n" : 0})


# # One - Hot Encoding

# In[97]:


home_ownership_new = pd.get_dummies(data["home_ownership"], drop_first = True)


# In[98]:


purpose_new = pd.get_dummies(data["purpose"], drop_first = True)


# In[99]:


initial_list_status_new = pd.get_dummies(data["initial_list_status"], drop_first = True)


# In[100]:


data = data.drop(["home_ownership","purpose", "initial_list_status"], axis = 1)


# In[101]:


data = pd.concat([data,home_ownership_new], axis = 1)


# In[102]:


data = pd.concat([data, purpose_new], axis = 1)


# In[103]:


data = pd.concat([data, initial_list_status_new], axis = 1)


# In[104]:


data.head()


# In[105]:


data = data.drop(["earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"], axis = 1)


# In[106]:


data.head()


# # Splitting by Date Column

# In[107]:


X = data.loc[data['issue_d'] < '2015-07-01']
y = data.loc[data['issue_d'] >= '2015-07-01']


# In[108]:


X.shape,y.shape


# In[109]:


X_train = X.drop(['default_ind','issue_d'],axis = 1)
y_train = X.default_ind
X_test = y.drop(['default_ind','issue_d'],axis = 1)
y_test = y.default_ind


# # Feature selection using SelectKBest

# In[110]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func = chi2, k=20)
fit = bestfeatures.fit(X_train,y_train)

dfscore = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)

featureScore = pd.concat([dfcolumns,dfscore],axis=1)
featureScore.columns = ['Specs','Score']
featureScore.head()

print(featureScore.nlargest(20,'Score'))


# In[111]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV


# # Simple Logistic Regression

# In[112]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[113]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[114]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[115]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# Balancing Imbalanced DataSet

# In[116]:


import imblearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


# In[117]:


sm = SMOTE(sampling_strategy = 0.5, random_state = 2)


# In[118]:


X_train['term'] = X_train['term'].astype(float)
X_train['grade'] = X_train['grade'].astype(float)
X_train['verification_status'] = X_train['verification_status'].astype(float)
X_train['pymnt_plan'] = X_train['pymnt_plan'].astype(float)
X_train['MORTGAGE'] = X_train['MORTGAGE'].astype(float)
X_train['NONE'] = X_train['NONE'].astype(float)
X_train['OTHER'] = X_train['OTHER'].astype(float)
X_train['OWN'] = X_train['OWN'].astype(float)
X_train['RENT'] = X_train['RENT'].astype(float)
X_train['credit_card'] = X_train['credit_card'].astype(float)
X_train['debt_consolidation'] = X_train['debt_consolidation'].astype(float)
X_train['educational'] = X_train['educational'].astype(float)
X_train['home_improvement'] = X_train['home_improvement'].astype(float)
X_train['house'] = X_train['house'].astype(float)
X_train['major_purchase'] = X_train['major_purchase'].astype(float)
X_train['medical'] = X_train['medical'].astype(float)
X_train['moving'] = X_train['moving'].astype(float)
X_train['other'] = X_train['other'].astype(float)
X_train['renewable_energy'] = X_train['renewable_energy'].astype(float)
X_train['small_business'] = X_train['small_business'].astype(float)
X_train['vacation'] = X_train['vacation'].astype(float)
X_train['wedding'] = X_train['wedding'].astype(float)
X_train['w'] = X_train['w'].astype(float)


# In[119]:


X_train_smote, y_train_smote = sm.fit_resample(X_train.astype('float') ,y_train)


# In[120]:


from collections import Counter
print("Before SMOTE :" , Counter(y_train))
print("After SMOTE :" , Counter(y_train_smote))


# # Logistic Regression after Balancing

# In[121]:


logreg = LogisticRegression()
logreg.fit(X_train_smote, y_train_smote)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[122]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[123]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[124]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Logistic Regression with GridSearch

# In[125]:


clf_logistic = LogisticRegression(solver='lbfgs')


# In[126]:


clf_logistic.fit(X_train_smote, y_train_smote)


# In[127]:


preds = clf_logistic.predict_proba(X_test)
preds_data = pd.DataFrame(preds[:,1], columns = ['prob_default'])
preds_data['loan_status'] = preds_data['prob_default'].apply(lambda x: 1 if x > 0.65 else 0)


# In[128]:


#print(confusion_matrix(y_test, preds_data['loan_status']))
print(accuracy_score(y_test,preds_data['loan_status']))
print(classification_report(y_test, preds_data['loan_status']))


# In[129]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, preds_data['loan_status'])
fpr, tpr, thresholds = roc_curve(y_test, preds_data['loan_status'])
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Random Forest

# In[130]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# In[131]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[132]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[133]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
rf_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('RF_ROC')
plt.show()


# # Naive Bayes

# In[134]:


# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train_smote, y_train_smote) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[135]:


#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # SVM with balancing

# In[136]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train_smote, y_train_smote)


# In[137]:


y_pred = svclassifier.predict(X_test)


# In[138]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # kNN without Balancing

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # kNN with Balancing

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_smote, y_train_smote)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




