import pandas as pd 
import datetime
import numpy as np 

######################################################################
########## Code to create the Datamart ###############################
########### The Datamart is named df3 ################################
######################################################################

#reading the data
card = pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\card.asc', sep=';')

disp = pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\disp.asc', sep=';')

account =pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\account.asc', sep =';')

district = pd.read_csv ('C:\\Users\\pdundon\\Desktop\\GroupPython\\district.asc', sep =';')

district_df =district.rename(columns={'A1': 'district_id', 'A2': 'district_name', 'A3': 'region', 'A4': 'pop_size', 'A5': 'no_of_municip<499', 'A6': 'no_of_municip<2000', 'A7': 'no_of_municip<10000 ', 'A8': 'no_of_municip>10000', 'A9': 'no_cities', 'A10': 'ratio_urban', 'A11': 'avg_salary', 
                         'A12': 'unemploymant_95', 'A13': 'unemploymant_96', 'A14': 'no_of_enterpreneurs', 'A15': 'no_crimes_95', 'A16': 'no_crimes_96'})

client = pd.read_csv ('C:\\Users\\pdundon\\Desktop\\GroupPython\\client.asc', sep =';')

loans = pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\loan.asc', sep = ';')

orders = pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\order.asc', sep = ';')

transactions = pd.read_csv('C:\\Users\\pdundon\\Desktop\\GroupPython\\trans.asc', sep = ';')
#-------------------------------------------------------------------------------------------------
#Cleaning Client's table

# defining a function that returns the middle two digits of a six digit integer to be used to extract customer's birthdate.
def get_mid2_digits(x):
    return int(x/100) % 100
    print(x)

# defining a function that returns the month of birth_number for male & female customers.
def get_month(x):
    mth = get_mid2_digits(x)
    if mth > 50:
        return mth - 50
    else:
        return mth

# defining a function that returns the month of the birth_number.
def get_day(x):
    return x % 100
    print(x)

# defining a function that returns the year of birth_number.
def get_year(x):
    return int(x/10000)

# defining a function that returns the gender from the client's birth_number.
def get_gender(x):
    mth = get_mid2_digits(x)
    if mth > 50:
        return 'F'
    else:
        return 'M'

# defining a function that converts the birth_number into a date.
def convert_int_to_date(x):
    yr = get_year(x) + 1900
    mth = get_month(x)
    day = get_day(x)
    return datetime.date(yr, mth, day)

# initialise end_date variable
end_date = datetime.datetime(2000,1,1)
end_date2 = datetime.date(2000,1,1)

# define function function to convert a date to age at end_date.
def convert_to_age_days(x):
    td = end_date - x
    return td.days

# defining a function that converts birth_number into age.
def convert_birthday_to_age(x):
    year = get_year(x) + 1900
    month = get_month(x)
    day = get_day(x)
    return convert_to_age_days(datetime.datetime(year,month,day))/365

#reference dates
start_date = datetime.date(1993,1,1)
end_date2 = datetime.date(2000,1,1)

def convert_date_to_days(x):
    td = x - start_date
    return td.days

# applying functions and rounding age    
client['client_age'] = client['birth_number'].map(convert_birthday_to_age)
client['client_age'] =client['client_age'].round(decimals=0)
client['client_gender'] = client['birth_number'].map(get_gender)

# building agegroups
client['age_group'] = client['client_age'] // 10 * 10

# deleting client['birth_number']
del client['birth_number']

# converts the account_date into a date.
account[['date']]= account['date'].map(convert_int_to_date)

#defining function to change frequency statement values to english
def acc_rename(frequency):
    
    if frequency == 'POPLATEK MESICNE':
        return 'monthly issuance'
    elif frequency == 'POPLATEK TYDNE':
        return   'weekly issuance'
    elif frequency == 'POPLATEK PO OBRATU':
        return 'issuance after transaction'
    else: 
        return 'unknown'
account[['frequency']]= account['frequency'].map(acc_rename)

# renaming column names in order to avoid confusion
account =account.rename(columns={'frequency': 'statement_freq','date': 'account_date' })

#changing column names in order to avoid duplicates
card.columns = ['card_id','disp_id','card_type', 'card_issued'] 
disp.columns = ['disp_id','client_id','account_id', 'disp_type']
#-------------------------------------------------------------------------------------------------
#extracting the year and month from card_iussued
card['card_issued'] = pd.to_datetime(card['card_issued'])
card['card_issue_year'] = card['card_issued'].dt.year
card['card_issue_month'] = card['card_issued'].dt.month
#-------------------------------------------------------------------------------------------------

#Preparing loans data
#convert loan date to integer
loans['date'] = loans['date'].map(convert_int_to_date)
loans['loan_month'] = pd.DatetimeIndex(loans['date']).month
loans['loan_year'] = pd.DatetimeIndex(loans['date']).year
del loans['date']

#Preparing orders data
#Translating k_symbol variable values from Czech to English
def convert_k_symbol_to_eng(x):
    if x == 'POJISTNE':
        return 'INSURANCE_PAYMENT'
    elif x == 'SIPO':
        return 'HOUSEHOLD_PAYMENT'
    elif x == 'LEASING':
        return 'LEASING_PAYMENT'
    elif x == 'UVER':
        return 'LOAN_PAYMENT'
    else:
        return 'UNKNOWN'
    
orders['order_k_symbol'] = orders['k_symbol'].map(convert_k_symbol_to_eng)
del orders['k_symbol']
del orders['account_to']

orders.head()

orders2 = orders.pivot(columns='order_k_symbol', index = 'order_id',values='amount')
orders2 = orders2.assign(order_id=orders2.index.get_level_values('order_id'))
sample= orders.merge(orders2, how = 'inner', on ='order_id')
del sample['order_k_symbol']
del sample['bank_to']
sample2 = sample.groupby('account_id', as_index=False).sum()
del sample2['order_id']
sample2 =sample2.rename(columns={'HOUSEHOLD_PAYMENT': 'ord_HOUSEHOLD_PAYMENT', 'INSURANCE_PAYMENT': 'ord_INSURANCE_PAYMENT', 'LEASING_PAYMENT': 'ord_LEASING_PAYMENT', 'LOAN_PAYMENT': 'ord_LOAN_PAYMENT', 'UNKNOWN': 'ord_UNKNOWN'})

tot_orders_amt = orders.groupby('account_id', as_index=False)['amount'].sum()
tot_orders_count = orders.groupby('account_id', as_index=False)['order_id'].count()
tot_orders_amt =tot_orders_amt.rename(columns={'amount': 'tot_order_amount' })
tot_orders_count =tot_orders_count.rename(columns={'order_id': 'tot_order_count' })

#Preparing trans data
#Translating trans type, operation and k_symbol from Czech to English
def convert_trans_type_to_eng(x):
    if x == 'PRIJEM':
        return 'CREDIT'
    elif x == 'VYDAJ':
        return 'WITHDRAWAL'
    else:
        return 'UNKNOWN'
    
def convert_trans_op_to_eng(x):
    if x == 'VYBER KARTOU':
        return 'CC_WITHDRAWAL'
    elif x == 'VKLAD':
        return 'CREDIT_IN_CASH'
    elif x == 'PREVOD Z UCTU':
        return 'COLLECTION_FROM_OTHER_BANK'
    elif x == 'VYBER':
        return 'WITHDRAWAL_IN_CASH'
    elif x == 'PREVOD NA UCET':
        return 'REMITTANCE_TO_OTHER_BANK'    
    else:
        return 'UNKNOWN'
    
def convert_trans_k_symbol_to_eng(x):
    if x == 'POJISTNE':
        return 'INSURANCE_PAYMENT'
    elif x == 'SLUZBY':
        return 'PAYMENT_FOR_STATEMENT'
    elif x == 'UROK':
        return 'INTEREST_CREDITED'
    elif x == 'SANKC. UROK':
        return 'SANCTION_INTEREST'
    elif x == 'SIPO':
        return 'HOUSEHOLD'
    elif x == 'DUCHOD':
        return 'OLD_AGE_PENSION'  
    elif x == 'UVER':
        return 'LOAN_PAYMENT'      
    else:
        return 'UNKNOWN'

transactions['trans_type'] = transactions['type'].map(convert_trans_type_to_eng)
transactions['trans_operation'] = transactions['operation'].map(convert_trans_op_to_eng)
transactions['trans_k_symbol'] = transactions['k_symbol'].map(convert_trans_k_symbol_to_eng)

del transactions['type']
del transactions['operation']
del transactions['k_symbol']
del transactions['account']

#Convert date variable to int
transactions['date'] = transactions['date'].map(convert_int_to_date)
transactions['trans_month'] = pd.DatetimeIndex(transactions['date']).month
transactions['trans_year'] = pd.DatetimeIndex(transactions['date']).year
del transactions['date']

tot_trans_amt = transactions.groupby('account_id', as_index=False)['amount'].sum()
tot_trans_count = transactions.groupby('account_id', as_index=False)['trans_id'].count()
tot_trans_amt =tot_trans_amt.rename(columns={'amount': 'tot_trans_amount' })
tot_trans_count =tot_trans_count.rename(columns={'trans_id': 'tot_trans_count' })

transactions.head()

#merging tables
merged = client.merge(district_df, how = 'left', on ='district_id')
temp = pd.merge(disp, card, how='left', on='disp_id')

temp2 = merged.merge(temp, how = 'inner', on ='client_id')
df = temp2.merge(account, how='left', on = 'account_id')

# del duplicate columns
del df['district_id_y']

# renaming final df columns
df =df.rename(columns={'district_id_x': 'district_id' })

df2 = df.merge(loans, how = 'left', on = 'account_id')

df3 = df2.merge(tot_trans_amt, how = 'left', on = 'account_id')
df3 = df3.merge(tot_trans_count, how = 'left', on = 'account_id')
df3 = df3.merge(tot_orders_amt, how = 'left', on = 'account_id')
df3 = df3.merge(tot_orders_count, how = 'left', on = 'account_id')
df3 = df3.merge(sample2, how = 'left', on = 'account_id')

#Add LOS variable (length of subscription)
df3['LOS'] = round((((end_date2) - df3['account_date']).dt.days)/365.25)

#Creating Target Variable 1 if status is good(A, C), 0 if bad (B, D), NA if not known 
df3.loc[df3.status =='A', 'Target'] = 1 
df3.loc[df3.status =='C', 'Target'] = 1 
df3.loc[df3.status =='B', 'Target'] = 0
df3.loc[df3.status =='D', 'Target'] = 0

##############################################
#Plotting indivudually good customers (Target =1) and bad customers (Target =0) to gain insights
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12,8)

def histogram(df, col_name, bins):
    # look at loan_duration.
    plt.hist(df[col_name], alpha=0.5, label=col_name, bins=bins)
    plt.legend(loc='upper right')
    plt.show()
  
def barchart(df, col_name):
    df[col_name].value_counts().plot(kind='bar', subplots=False, color = 'g', alpha = 0.75)

def boxplot(df, col_names, by=None):
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)
    df.boxplot(column=col_names, return_type='axes', by=by)    

#Quick view of number of targets vs non-targets among loan customers   
barchart(df3, 'Target')

#Creating a dataframe for bad loans, dataframe for good loans
df_bad = df3[df3.Target == 0]
df_good = df3[df3.Target == 1]

#Good vs Bad loans by loan amount
histogram(df_bad, 'amount_x', 10)
histogram(df_good, 'amount_x', 10)

#Good and bad loans by gender
barchart(df_bad, 'client_gender')
barchart(df_good, 'client_gender')

df_good['client_gender'].value_counts()

#Good and bad loans by avg_salary
histogram(df_bad, 'avg_salary', 10)
histogram(df_good, 'avg_salary', 10)

#Good and bad loans by length of subscription
barchart(df_bad, 'LOS')
barchart(df_good, 'LOS')
histogram(df_bad, 'LOS', 7)
histogram(df_good, 'LOS', 7)

#Good and bad loans by credit card type (small sample size)
barchart(df_bad, 'card_type')
barchart(df_good, 'card_type')

#Good and bad loans by statement frequency
barchart(df_bad, 'statement_freq')
barchart(df_good, 'statement_freq')

#Good and bad loans by geographical district
barchart(df_bad, 'district_name')
barchart(df_good, 'district_name')

#Good and bad loans by geographical region
region_bad = (df_bad.groupby('region')['loan_id'].count())
region_bad.plot.bar(color='green')
plt.ylabel('Number of bad loans', fontsize = 15)
plt.xlabel('Region', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Bad loans per region', fontsize = 15)
plt.show()
region_good = (df_good.groupby('region')['loan_id'].count())
region_good.plot.bar(color='green')
plt.ylabel('Number of good loans', fontsize = 15)
plt.xlabel('Region', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Good loans per region', fontsize = 15)
plt.show()

df_good['region'].value_counts()
df_bad['region'].value_counts()

#Good and bad loans by age group
barchart(df_bad, 'age_group')
barchart(df_good, 'age_group')

df_good['age_group'].value_counts()

histogram(df_bad, 'age_group', 7)
histogram(df_good, 'age_group', 7)

#Good and bad loans by total amount of account transactions
histogram(df_bad, 'tot_trans_count', 10)
histogram(df_good, 'tot_trans_count', 10)

df_good['tot_trans_count'].mean()
df_bad['tot_trans_count'].mean()

#Good and bad loans by year of loan taken out
barchart(df_bad, 'loan_year')
barchart(df_good, 'loan_year')

df_good['loan_year'].value_counts()
df_bad['loan_year'].value_counts()

#Good and bad loans by loan duration (months)
duration_bad = (df_bad.groupby('duration')['loan_id'].count())
duration_bad.plot.bar(color='green')
plt.ylabel('Number of bad loans', fontsize = 15)
plt.xlabel('Duration of loan (months)', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Bad loans duration distribution', fontsize = 15)
plt.show()
duration_good = (df_good.groupby('duration')['loan_id'].count())
duration_good.plot.bar(color='green')
plt.ylabel('Number of good loans', fontsize = 15)
plt.xlabel('Duration of loan (months)', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Good loans duration distribution', fontsize = 15)
plt.show()

df_good['duration'].value_counts()
df_bad['duration'].value_counts()

#Looking at good/bad customers and if they have a credit card
df_nocard = df3[df3.card_type.isnull()]
df_card = df3[df3.card_type.notnull()]
df_nocard = df_nocard[df_nocard.status.notnull()]
df_card = df_card[df_card.status.notnull()]

df_nocard['Target'].value_counts()
df_card['Target'].value_counts()

nocc = (df_nocard.groupby('Target')['loan_id'].count())
nocc.plot.bar(color='green')
plt.ylabel('Number of customers', fontsize = 15)
plt.xlabel('Customer status (Target or non-target)', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Customer status with no credit card', fontsize = 15)
plt.show()
cc = (df_card.groupby('Target')['loan_id'].count())
cc.plot.bar(color='green')
plt.ylabel('Number of customers', fontsize = 15)
plt.xlabel('Customer status (Target or non-target)', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Customer status with credit card', fontsize = 15)
plt.show()

#----------------------------------------------------------------------------------------------
#preparing for variable selection
dftargets = df3[(df3['Target'] == 1) | (df3['Target'] == 0)]

del dftargets['district_name']
del dftargets['client_gender']
del dftargets['region']
del dftargets['disp_type']
del dftargets['card_type']
del dftargets['card_issued']
del dftargets['statement_freq']
del dftargets['account_date']
del dftargets['status']
del dftargets['unemploymant_95']
del dftargets['unemploymant_96']
del dftargets['no_crimes_95']
del dftargets['no_crimes_96']
del dftargets['card_id']
del dftargets['card_issue_year']
del dftargets['card_issue_month']

# Feature Importance with Extra Trees Classifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

#separate the target from the variables (has to be changed according to the number of variables)
X = dftargets.iloc[:,1:32]
Y = dftargets.iloc[:,32]

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

#create barchart with importance
results = model.feature_importances_
y_pos = np.arange(len(results))
variables = (dftargets.columns.values)
plt.bar(y_pos,results,color='black')
plt.xticks(y_pos, variables, rotation = 90)
plt.savefig('C:/Users/Remo/0.png')
plt.show()

#--------------------------------------------------------------------------------------------
# display some general insights for the beginning of the report
status = (df3.groupby('status')['loan_id'].count())
status.plot.bar(color='black')
plt.ylabel('Frequency', fontsize = 15)
plt.xlabel('Status', fontsize= 15)
plt.xticks(rotation = 45)
plt.title('Distribution of loan status', fontsize = 15)
plt.show()

age_group = df3.groupby('age_group')['client_id'].count()
age_group.plot.bar(color= 'black')
plt.ylabel('Number of people', fontsize=15)
plt.xlabel('Age Group', fontsize=15)
plt.title('Number of people per age group', fontsize = 15)
plt.show()

region = (df3.groupby('region')['avg_salary'].agg('mean').sort_values(ascending=False))
region.plot.bar(color='black')
plt.ylabel('Average salary', fontsize=15)
plt.xlabel('Region', fontsize=15)
plt.title('Average salary for different regions', fontsize=15)
plt.show()

cards = (df3.groupby('card_type')['card_type'].agg('count').sort_values(ascending=False))
cards.plot.bar(color = 'black')
plt.ylabel('Number of cards', fontsize=15)
plt.xlabel('Card type', fontsize=15)
plt.title('Cards per type', fontsize=15)
plt.show()



##############################################

import matplotlib.pyplot  as plt

#Subsetting data to include only targets
targets=df3.loc[df3['Target'].isin(['0','1'])]
targets_bad = df3[df3.Target == 0]
targets_good = df3[df3.Target == 1]

#######creating a scatterplot to show relationships between loan year and loan amount for targets/non-targets
plt.scatter(targets.loan_year, targets.amount_x, c=targets.Target,cmap='flag')
plt.title("Total Amount of  Loan as a function of Year", font_scale=2.5)
plt.show()


#######creating a scatterplot to show relationships between loan payments and loan amount for targets/non-targets
plt.scatter(targets.payments, targets.amount_x, c=targets.Target,  cmap='flag' )
plt.title("Amount of Monthly Loan Payment as a function of Total Loan Amount", font_scale=2.5)
plt.show()
 

#######histogram to show how many loans per year
hist = targets.hist(bins=5,column='loan_year', color='grey', grid= False)
plt.xlabel('Year')
plt.ylabel('Count of Loans per Year')


#############################################
import seaborn as sns
sns.set(style="whitegrid")

# Draw a nested barplot to show total loan amount for targets/non-targets and sex
g = sns.catplot(x="Target", y="amount_x", hue="client_gender", data=targets,
                height=6, kind="bar", palette="Blues_d")
g.despine(left=True)
g.set_ylabels("Total Amount of Loan")
plt.title("Total Amount of Loan per Target Group & Gender", font_scale=2.5)


# Drawing a nested barplot to show differences in amounts of loan payment for Targets/non-targets and sex
g = sns.catplot(x="Target", y="payments", hue="client_gender", data=targets,
                height=6, kind="bar", palette="Blues_d")
g.despine(left=True)
g.set_ylabels("Amount of Monthly Loan Payment")
plt.title("Amount of Monthly Loan Payment per Target Group & Gender", font_scale=2.5)

# Drawing a nested barplot to show Total Order Amount for Targets/non-targets and sex

g = sns.catplot(x="Target", y="tot_order_amount", hue="client_gender", data=targets,
                height=6, kind="bar", palette="Blues_d")
g.despine(left=True)
g.set_ylabels("Total Order Amount")
plt.title("Total Order Amount per Target Group & Gender", font_scale=2.5)


# Drawing a nested barplot to show LOS for Targets/non-targets and sex

g = sns.catplot(x="Target", y="LOS", hue="client_gender", data=targets,
                height=6, kind="bar", palette="Blues_d")
g.despine(left=True)
g.set_ylabels("Length of Relationship")
plt.title("Length of Relationship per Target Group", font_scale=2.5)


#########################################

#Boxplot showing monthly loan payments for different Target Groups 
boxplot = targets.boxplot(column='payments', by='Target',  grid=False, fontsize=12)
plt.xlabel('Target Group')
plt.ylabel('Monthly Loan Payments')

#Boxplot showing monthly loan order payments for different Target Groups 
boxplot = targets.boxplot(column='ord_LOAN_PAYMENT', by='Target',  grid=False, fontsize=12)
plt.xlabel('Target Group')
plt.ylabel('Loan Payment Orders')

#Boxplot showing monthly household order payments for different Target Groups 
boxplot = targets.boxplot(column='ord_HOUSEHOLD_PAYMENT', by='Target',  grid=False, fontsize=12)
plt.xlabel('Target Group')
plt.ylabel('Household Payment Orders')

#Boxplot showing total order amounts for different Target Groups 
boxplot = targets.boxplot(column='tot_order_amount', by='Target',  grid=False, fontsize=12)
plt.xlabel('Target Group')
plt.ylabel('Total Order Amount')


##Parts of the code were retreived from the Financial Programming course materials ##