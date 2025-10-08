## This script will aim to generate some basic summary statistics about a data science sallary dataset and explore insights.
## Do I know what I'm doing? Not really. But let's see how it goes.

## importing modules that are probably just generically useful for most data analysis scripts.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

## Now it's time to get that data...
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "data_science_job_posts_2025 (2).csv"

df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"sidraaazam/data-science-roles-skills-and-salaries-2025",file_path)

## now the data is loaded into a variable called "df" (pandas dataframe).
## I think now what I'll want to do is get some basic summary statistics about the dataset.

## I'll start with salaries. After taking a look at the dataset, I've realized that salaries are all presented in '€'(thankfully).
## Some are presented as single values, while others are ranges. I'd like to strip out the '€' symbols, commas, and convert ranges to averages.
## I'll define a function that will clean this up for me.

def salary_clean(salary):
    if pd.isna(salary):
        return np.nan
    salary = salary.replace('€', '').replace(',', '').strip()
    if '-' in salary:
        low, high = salary.split('-')
        return (float(low) + float(high)) / 2
    else:
        try:
            return float(salary)
        except ValueError:
            return np.nan

## Now I'll apply this function to the 'Salary' column in the dataframe.
df['Cleaned_Salary'] = df['salary'].apply(salary_clean).astype(float)

## Now that I have cleaned up salary info, I think maybe we'll make a histogram of the salaries.
df['Cleaned_Salary'].plot.hist(bins=30, edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Distribution of Data Scientist Salaries')
plt.grid(True)
plt.show()  

## That plot doesn't actually look that good, would be better if the salaries were shown in more human-readable number format.
## For now I think it will do. At least we are able to see the overall shape of salary distribution.