
## Introduction 

This project aims to investigate the characteristics of good and bad clients for a Czech bank. This report will aid bank managers to better understand what makes a good client and guide their subsequent decision-making by offering additional services only to ‘good customers’. At the same time, the analysis will help bank managers to identify ‘bad customers’ that are more likely to not pay their loans and thus lead to bank losses. To this end, information from various datasets of the bank has been combined to create a single Datamart. The datasets included accounts, transactions, clients, orders, loans, districts, deponents, demographics, credit cards. 

## Variable Definition 


* client_id: client identifier 

* district_id:	district identifier

* client_age:	age on the 1.1.2000 based on the birthday entry

* client_gender: gender of the client

* age_group: 
  * agegroup 10: age ranging from 00-10 
  * agegroup 10: age ranging from 11-19
  * agegroup 20: age ranging from 20-29
  * agegroup 30: age ranging from 30-39
  * agegroup 40: age ranging from 40-49
  * agegroup 50: age ranging from 50-59
  * agegroup 60: age ranging from 60-69
  * agegroup 70: age ranging from 70-79
  * agegroup 80: age ranging from 80-89
  
* district_name: name of the district

* region:	region in the Czech Republic

* pop_size:	population size of the region

* no_of_municip<499: no. of municipalities with inhabitants <499

* no_of_municip<2000:	no. of municipalities with inhabitants 500-1999

* no_of_municip<10000: no. of municipalities with inhabitants 2000-9999

* no_of_municip>10000: no. of municipalities with inhabitants 10000+

* no_cities: number of cities

* ratio_urban: ratio of urban inhabitants

* avg_salary: average salary 

* unemployment_95: unemployment rate in 1995

* unemployment_96: unemployment rate in 1996

* no_of_entrepreneurs: number of entrepreneurs

* no_crime_95: number of crimes in 1995

* no_crime_96: number of crimes in 1996

* disp_id: disposition to an account

* account_id:	identification of the account

* disp_type: owner or Disponent

* card_id: card identifier

* card_type: card type (junior, classic or gold)

* card_issued: date when the card was issued

* card_issue_year: year in which the card was issued

* card_issue_month:	month in which the card was issued

* statement_freq: frequency of issuance of

* statements: monthly issuance, weekly issuance or issuance after transaction

* account_date: date on which the account was opened

* loan_id: loan identifier

* amount_x: amount of the loan

* duration: duration of the loan

* payments:	monthly payments

* status:	status of paying off the loan
  * 'A' stands for contract finished, no problems,
  * 'B' stands for contract finished, loan not payed,
  * 'C' stands for running contract, OK so far,
  * 'D' stands for running contract, client in debt
  
* loan_month: month in which the loan was issued

* loan_year: year in which the loan was issued

* tot_trans_amount: total amount of transactions

* tot_trans_count: numbers of transactions

* tot_order_count: number of orders made

* amount_y:	total amount of orders

* ord_Houshold_payment:	total amount of orders for household payments

* ord_insurance_payment: total amount of orders for insurance payments

* ord_leasing_payment: total amount of orders for leasing payments

* ord_loan_payment:	total amount of orders for loan payments

* ord_unknown_payment: total amount of orders for unknown payments

* LOS: length of relationship in years

* Target: nan for people without loan
  1 for people with status A or C
  0 for people with status B or D
  
  ## General Data Overview & Insights

* From all the clients in the Datamart only 827 people have a loan. 751 are clients rated A or C (positive) and 76 are rated either B or D (negative).

* Looking at the different age groups, we notice that most clients are between 20 and 60 years old. The age groups 20, 30, 40 and 50 all consist of around 1000 clients. On the other hand, only a few clients are below 20 or above 80 years old. 

* Regarding the average salary for the different regions, it is obvious that in Prague salaries are much higher compared to other regions as it is the capital of the country. The average salary for the other regions lies at around 9000 with only minimal differnces between regions.

* Only around 1 out of 6 customers in the database owns a credit card. Most  owners own a credit card of type “classic” whereas the types “junior” and “gold” are not as common.

 ## Specific Insights
  
* The importance of different variables for deciding whether a customer will pay his/her loan was calculated  using an Extra-Tree-Classifier. The variables “Total order Amount” and “Duration” have the highest influence whereas the “ord Household Payment” and “ord Insurance Payment” have almost no influence. 

*Note that some of the variables of the original Datamart were excluded for reasons such as correlation or not numeric.

## Classifying Good vs Bad Customers

As established, clients whose loan status was either ‘A’ or ‘C’ were classified as ‘Good clients’ and those whose loan status was either ‘B’ or ‘D’ were classified as ‘Bad clients’. ‘Good clients’ were represented by a value of 1 for our target variable and ‘Bad clients’ by 0.

## General Trends

* First,  the distribution of the loans for the different years of interest was investigated. An increasing trend was observed in that more loans are issued each year during the period examined, from 1993 to 1998. In particular, the bank issued less than 50 loans in 1993 as opposed to 1998, with the number of loans exceeding 400.

* An examination on how the duration of the loan affects its final status revealed that that 12-month loans accounted for 20% of good loans, and only 14.5% of bad loans. Conversely, 24-month loans were the worst performing loan duration, with 22% of those loans defaulting and 20% turning out well. 

* Another useful insight is that ‘Good clients’ tend to make fewer permanent orders associated with Loan Payments. It is also worth noting that these clients also make more household-related payment orders. Conversely, bad clients are related almost to no household permanent orders. Specifically, ‘Good clients’ make on average orders of 7266 euros which are related to household payments, whereas the amount that bad clients’ household orders is close to 0 (with some outliers).

* Interestingly, 'Good clients' are associated with greater amounts of total permanent orders. Particularly, mean total order amount for ‘Good clients’ is 9351 euros (including males and females) compared to 7641 euros for their non-targets counterparts. From this it can be inferred that ‘Good clients’ tend to make permanent orders of greater amounts. Therefore, the bank should focus on these clients as they are more likely to lead to bank profits.

* In addition, ‘Good clients’, both male and female, tend to have lower monthly loan payments. In stark contrast, clients that do not pay their loans on time tend to have higher monthly loan payments. This is particularly true for the female group. Therefore, managers should closely monitor these clients to better handle loan payments and hence, subsequently reduce bank losses to the minimum. Similarly, ‘Bad clients’ obtain loans of greater amount than those of 'Good clients'. This trend is more pronounced for the female group with loans amounting to 225.000 (currency not defined) as opposed to 150.000 in the target group.

* However, variables like the length of a client’s relationship (LOS) with the bank do not seem to impact the overall status of the loan. There is no great difference between Target groups (i.e. Good and Bad clients) regarding the LOS with the bank. In fact, good clients who tend to pay their loans on time have a LOS of approximately 4 years, slightly shorter than that of  the ‘Bad clients’ which amounts to 4.5.

* Some areas contribute more to bad loans than good, and vice versa. Of note, Brno, the 2nd most populated city in the Czech Republic, accounted for only 3.3% of good loans, and doubled its representation for bad loans (6.5%). Conversely, Ústí nad Orlicí district only had one bad loan customer compared to 22 good loan customers, providing the best good to bad loan ratio. 

* North Moravia is the worst performing region overall, accounting for 25% of bad loans and only 16% of good loans. Prague region, the largest in the Czech Republic, provides one best ratios of good loans (12%) to bad loans (9%), alongside North Bohemia (9% good to 1% bad). 

* The average of total account transactions for bad loan customers was quite higher than those who were good loan customers (299 transactions versus 280 transactions respectfully), so can infer that those who default on their loans are on average more active with their accounts.

* We also evaluated customers who took out a loan to see if they also held a credit card issued by the bank. Of those customers, 170 loan customers held a credit card and 657 did not. Among the 170 credit card holders, 97% were good loan customers and 3% were not. Among the 657 non-credit card holders, 89% were good loan customers and 11% were not. We can therefore infer that customers who took out loan and hold a credit card from the bank have a better chance to be a good customer.

* Finally, as the years increase more loans and of greater amount are issued. Interestingly, the year 1998 was a good year for the bank with only 4 ‘Bad clients’, who failed to pay their loans on time. It is worth noting that although ‘Bad clients’ are slightly more concentrated in the higher end of the y-axis, confirming the previous finding that ‘bad loans’ tend to be of greater amounts.


* After analysis, we see that the most contributing variables in determining loan performance (and a good customer vs a bad customer). Therefore, we can start to build an introductory profile of good and bad customers as seen in this table: 

## **Summary **

* Good Loans/Customers	

Higher loan amount, longer loan time-period, may reside in Brno or North Maravia, are senior-aged, likely do not hold a credit card issued by the bank

*  Bad Loans/Customers	 
Lower loan amount, lower loan time-period, may reside in Prague region, are middle-aged, may also hold a credit card issued by the bank, greater amounts in total order payments, more household-related permanent orders	


## **References **

PKDD'99 Discovery Challenge: Guide to the Financial Data Set. 16 Jan. 2018, lisp.vse.cz/pkdd99/berka.htm.





