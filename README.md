# Automation Local Food Bank
Building AI course project

## Summary
First phase: Creation of machine learning models based on historical data from a Food Bank.

Second phase: Automation of a local Food Bank, using an App integrated with a ChatBot with NLP, to manage the food demand of needy families.

![image of a Food Bank](/alfb000.jpg)

## Background
A food bank is a volunteer-based organization whose objective is to recover food, especially non-perishable food, from companies, supermarkets and individuals, through collaborating associations, for subsequent free distribution to people at risk of social exclusion. 

The management of a food bank is complex, so sometimes resources are wasted (food that expires, that remains in stock, that is not distributed efficiently, etc.) and it is complicated to organize the volunteers, the inventory in the warehouses and the collection and distribution of food.

I believe that, in order to tackle this project, it would be necessary to do it in two stages: 

First stage: With the historical data that we currently have in spreadsheets, we can make a prediction of the food needed for distribution in each campaign, through automatic training, to improve the use of resources, avoiding the accumulation of surplus in warehouse, the expiration of food or the increase of spoiled or damaged food for various reasons, making a more efficient food distribution.

Second stage: Once the results of the first stage have been seen, in order to improve the system, an integrated food management system could be implemented in which a specialized App integrated with a chatbot would play an important role, using predictive data from automatic training based on historical data on inventory, storage, requests, food distribution and volunteer management; as well as communication with the parties involved with valuable information (recipients, donors, neighborhood associations, volunteers, etc.).

My personal motivation comes from the voluntary participation in the work of a food bank. This issue is important because it is about optimizing resources for families living in poverty or at risk of social exclusion.

In the project I am presenting for the Building AI course, I will develop only the first stage.

## Data and AI techniques

The data sources are based on the historical data collected from previous campaigns for food collection and distribution: lists of food collected or purchased, lists of food distributed, lists of beneficiary associations, lists of volunteers, etc. These data are currently collected in spreadsheets, without personal data. 

In this first phase, machine learning techniques will be applied to predict needs in future campaigns based on data collected in previous campaigns.
 
To feed the machine learning models for the predictions and functionalities mentioned above, accurate and relevant user data collection is essential. 

In principle, we will start from the data currently available, provided by the neighborhood associations in different campaigns and by the annotations made in spreadsheets. 

The data collected, for a first study for forecasting future needs, are contained in a standardized model with a series of interrelated tables.
In order to be able to apply a machine learning model, a data integration has been performed, combining the tables by means of union operations (data join), in order to be able to work with a single denormalized table, with a previous non-exhaustive data cleaning, due to the fact that there were hardly any missing values or errors. 

The names of the different categories or variables are in Spanish language, because the project is applied with data collected locally, in Spain.
Although in a second phase of the project we will be able to have more quality data over time, the data available at this time are as follows:
1. neighborhood (barriada) indicative: Each food collection and distribution campaign is done by neighborhoods, with the help of the neighborhood associations.
3. Campaign (campana): Currently there are four annual campaigns that coincide approximately with the seasons of the year.
4. Population from zero to two years old (bebes): babies between zero and two years old in each zone or neighborhood.
5. Population from 3 to 10 years old (ninos): Population between three and ten years old.
6. Population aged 11 to 65 years (adultos): Population aged between eleven and sixty-five years.
7. Population over 65 years old (seniors).
8. Unemployment rate (desempleo): Unemployment rate by neighborhood in percent.
9. Users served (usuarios): Number of people served in each neighborhood and campaign.
10. Expiry of food type A01 (ac01= expired food type A01): Percentage of that food A02 expired (not deliverable).
11. Distributed food type A01 (ad01 = distributed food type A01): Quantity of food type coded A01 distributed in units (packages, cans, UTH). All foods are packaged, canned or bottled, and each package was noted as a unit, regardless of weight or size.

Food types are coded as follows:

Ad01 = Baby food (infant formula, infant milks, etc.) - Baby food (infant formula, infant milks, etc.).

Ad02 = Cereals and grains (rice, pasta in various forms, quinoa, flaked or grained oats, buckwheat, millet, breakfast cereals, whole grain cereals)

Ad03 = Dried pulses (lentils, chickpeas, beans, dried peas, soybeans)

Ad04 = Canned and tinned foods (canned vegetables, canned fruits, canned fish, canned meats, peanut butter or chocolate, powdered milk, condensed milk, honey)

Ad05 = Nuts and seeds (almonds, walnuts, pumpkin seeds, sunflower seeds, etc.)

Ad06 = Oils and fats (olive oil, sunflower oil, coconut oil, canola oil, butters)

Ad07 = Sweets and snacks (granola bars, cookies, chocolate bars, dried fruits, jams)

Ad08 = Sauces and condiments (salt, ketchup, mustard, soy, bouillon cubes or powdered broth)

Ad09 = Beverages (coffee, tea, powdered beverages, UTH bedding, pasteurized fruit juices, packaged soups)

Ad10 = Flours and bakery products (wheat flour, corn flour, toasted bread, packaged sliced bread)
    
Given the assumption that different food types are significantly influenced by the input characteristics and for interpretability reasons, I will choose to treat each food type separately, trying different models, taking care not to over-fit. So we will have as many tables as different types of food.

Although, at first glance, it can be deduced that the seasonal variations of the different campaigns influence the amount of food donated and requested, by having a temporal characteristic (campna) that practically coincides with the seasons of the year, the model chosen can recognize the patterns associated with each period, which allows not to make further divisions and to treat in a single file each type of food.

Attached are files in CSV format, on which machine learning algorithms based on several models will be applied to study which ones respond better to this data set.

I will start with multiple linear regression model, using as dependent variable the amount of food to distribute (adxx for each type of food).

This model is appropriate because you want to know how several independent variables (predictors) influence a continuous dependent variable (the amount of food to be distributed in 'adxx'). In addition, the multiple linear regression model can handle both numerical variables (such as babies, children, adults, seniors, unemployment, users, ac01) and categorical variables (such as neighborhood and bell), although categorical variables must be properly coded before including them in the model.

```
# Multiple linear regression

# The necessary pandas and sklearn libraries are imported.
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load the data from the CSV file from the specified path.
data_path = 'https://github.com/marmartiher/ALFB/blob/main/NBancoAlimentos.csv'
data = pd.read_csv(data_path)

# Prepare the data by separating the independent variables (X) from the dependent variable (y).
X = data.drop('ad01', axis=1)  # Independent variables
y = data['ad01']  # Dependent variable

# Coded the categorical variables 'slum' and 'bell' as One-Hot, and took the numeric variables unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['barriada', 'campana'])
    ], remainder='passthrough')

# A pipeline including the preprocessing and the linear regression model is created.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data ensuring all 'campana' groups are represented in the test set
# This is done to capture the annual variability and the particularity of the winter or Christmas campaign
# Since 'campana' is a categorical variable, we will use a splitting approach that maintains the proportion of each category

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['campana'], random_state=42)

# Fit the model using the training data: model_pipeline.fit().
model_pipeline.fit(X_train, y_train)

# Predictions are made on the test set y_pred.
y_pred = model_pipeline.predict(X_test)

# The mean squared error MSE of the predictions is calculated and displayed.
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# The mean absolute error MAE is calculated.
mae = mean_absolute_error(y_test, y_pred)

# The coefficient of determination (R^2) is calculated.
r2 = r2_score(y_test, y_pred)

# Display the values of MAE and R^2.
print(f'MAE: {mae}')
print(f'R^2: {r2}')

# The minimum and maximum of the dependent variable ad01 is calculated.
ad01_min = data['ad01'].min()
ad01_max = data['ad01'].max()

# Calculate the range (difference between the maximum and minimum)
ad01_range = ad01_max - ad01_min

# Calculate the mean of 'ad01'
ad01_mean = data['ad01'].mean()

# These previous values are displayed.
print(f"Min: {ad01_min}, Max: {ad01_max}, Range: {ad01_range}, Mean: {ad01_mean}")

# The necessary libraries for graphics are imported.
import seaborn as sns
import matplotlib.pyplot as plt

# Creating scatter plots
# Prepare a list of independent variable names, excluding 'ad01'.
independent_variables = data.columns.drop('ad01')

# For each independent variable, create a scatter plot with 'ad01'.
for var in independent_variables:
    sns.scatterplot(data=data, x=var, y='ad01')
    plt.title(f'Relationship between {var} and ad01')
    plt.show()

```

The values shown in this model are as follows:

MSE: 248.4674999045983

MAE: 11.28003393725271

R^2: 0.9525076234939224

Min: 112, Max: 442, Rango: 330, Media: 268.18548387096774

** Model evaluation considerations: **

The MSE is a measure of the quality of an estimator; it is always non-negative, and smaller values indicate a better fit. An MSE of 248.47 suggests that, on average, the model predicts the dependent variable ad01 with a mean square error of 248.47. Since the MSE depends on the scale of the variable, it is useful to compare it with the range and mean of ad01 to get a better idea of its magnitude.

The MAE is another measure of error that provides the mean absolute difference between observed and predicted values. An MAE of 11.28 indicates that, on average, the model predictions deviate by 11.28 units from the actual values. It is a more robust measure and easier to interpret compared to the MSE, especially in the presence of outliers.

El R^2 es una medida que indica la proporción de la variación en la variable dependiente que es predecible a partir de las variables independientes. Un R^2 de 0.95 es bastante alto, lo que sugiere que el modelo es capaz de explicar el 95% de la variabilidad observada en ad01, indicando un buen ajuste.

Min: 112, Max: 442, Range: 330, Mean: 268.19.  These statistics provide an overview of the distribution of the dependent variable ad01. The variation of ad01 between 112 and 442, with a range of 330, indicates a wide dispersion in the data. The mean of 268.19 reflects the average value of ad01 in the data set.

The relationship between the MSE and the rank of ad01 may offer additional insights. Although the MSE seems high, the rank of the dependent variable is also high. Therefore, an MSE of 248.47 can be considered reasonable in this context. Furthermore, the high value of R^2 suggests that the model is effective in predicting ad01 from the available independent variables.

The results indicate a fairly effective linear regression model for predicting ad01, with a fit that explains a large proportion of the variability in the dependent variable. Interpreting these results in the specific context of the data (e.g., the meaning of ad01, neighborhood, and bell) may provide further insights on how to improve the model.

Visualizing the relationship between each independent variable and ad01 using scatter plots can help identify linear or nonlinear patterns, potential outliers, and whether the relationship between variables is direct or inverse. This can be useful for future model adjustments or to identify variables that might need transformations.

We can see that, for example, the relationship between babies and the amount of food ad01, there is a positive relationship between the increase in babies and the increase in food type ad01. Which is logical in this case, since food type ad01 encompasses food intended for babies. The graph shows different groups of points, which suggests that there are subgroups within the data, but this is probably due to the fact that at certain times of the year (food distribution campaigns) there is usually more demand for food in general and, therefore, for this type of food as well.

![image of a Food Bank](/alfb001.jpg)
