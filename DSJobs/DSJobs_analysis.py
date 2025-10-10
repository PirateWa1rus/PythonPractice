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
## I'll omit the top three salaries to avoid skewing the histogram too much.
df_filtered = df[df['Cleaned_Salary'] < 600000]
#df_filtered['Cleaned_Salary'].plot.hist(bins=30, edgecolor='black')
#plt.xlabel('Salary')
#plt.ylabel('Frequency')
#plt.title('Distribution of Data Scientist Salaries')
#plt.grid(True)
#plt.show(block = False)  

## With the omission of outliers, I'm seeing what I believe to be two overlapping normal distributions.
## I'd like to see what the distribution of salaries is like when broken down by experience level.
## I'll make a boxplot for this. 

plt.figure(figsize=(10, 6))
sns.boxplot(x='seniority_level', y='Cleaned_Salary', data=df_filtered, showmeans=True, meanline=True)
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.title('Salary Distribution by Experience Level')
plt.grid(True)
plt.show(block = False)



## The boxplot suggests that "junior" and "midlevel" might not really be meaningful distinctions.
## I'll want to dig into that more later, probably by examining skills for each category. 
## For now, I think I'd like to make histograms of salaries by experience level all shown together in one window as separate plots.

experience_levels = df_filtered['seniority_level'].dropna().unique()
#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5), sharey=False)
#axes = axes.flatten()
#for ax, level in zip(axes, experience_levels):
#    sns.histplot(data=df_filtered[df_filtered['seniority_level'] == level], x='Cleaned_Salary', bins=30, kde=True, ax=ax)
#    ax.set_title(level)
#    ax.set_xlabel('Salary')
#    ax.set_ylabel('Density')

#plt.suptitle('Salary Distribution by Experience Level')
#plt.tight_layout()
#plt.show(block = False)



## These plots make it clearer that "junior" has far fewer records than the other categories, and distributions are not exactly normal.
## I'll try recategorizing seniority_level into just "junior" and "senior" (combining midlevel and senior) and see how that looks.
df_filtered['Seniority_Binary'] = df_filtered['seniority_level'].dropna().apply(lambda x: 'junior' if x == 'junior' or x == 'midlevel' else 'senior')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=False)
axes = axes.flatten()
for ax, level in zip(axes, ['junior', 'senior']):
    sns.histplot(data=df_filtered[df_filtered['Seniority_Binary'] == level], x='Cleaned_Salary', bins=30, kde=True, ax=ax)
    ax.set_title(level)
    ax.set_xlabel('Salary')
    ax.set_ylabel('Density')

plt.suptitle('Salary Distribution by Lumped Experience Level')
plt.tight_layout()
plt.show(block = False)

input('Press Enter to exit...')

## These are actually pretty interesting results. We can pretty clearly see that the "junior" category is not normally distributed.
## The "Senior" category is closer to normal, but still has a bit of a right skew.
## I'll be curious to see if particular skills are associated with identifiable distribution patterns.

## To get started, I'll need to learn something about what skills are listed.
## I took a look at the skills data, and I'm seeing that most entries are comma-separated lists of skills.
## What I'd like to do is understand which individual skills appear most often for 'junior' roles. I'll look at 'senior' roles later.
from collections import Counter
jr_skill_counter = Counter()
for skills in df_filtered[df_filtered['Seniority_Binary'] == 'junior']['skills'].dropna():
    skill_list = [skill.strip(" []'\"").lower() for skill in skills.split(',') if skill.strip(" []'\"")]
    jr_skill_counter.update(skill_list)
most_common_jr_skills = jr_skill_counter.most_common(25)
#print("Most common skills for junior roles:")
#for skill, count in most_common_jr_skills:
#    print(f"{skill}: {count}")

## Okay... that was a bit of a doozy (is that how you spell that word?). I had some issues with stripping characters effectively, but I got there.
## I suppose now it would makes sense to just modify that block to make a list also for senior roles.
sr_skill_counter = Counter()
for skills in df_filtered[df_filtered['Seniority_Binary'] == 'senior']['skills'].dropna():
    skill_list = [skill.strip(" []'\"").lower() for skill in skills.split(',') if skill.strip(" []'\"")]
    sr_skill_counter.update(skill_list)
most_common_sr_skills = sr_skill_counter.most_common(25)
#print("Most common skills for senior roles:")
#for skill, count in most_common_sr_skills:
#    print(f"{skill}: {count}")

## Now we can compare the top skills for both levels.
print(f"{'No.':<5} {'Junior Skill':<20} {'Count':<7} {'Senior Skill':<20} {'Count':<7}")
print("-" * 70)
for i in range(max(len(most_common_jr_skills), len(most_common_sr_skills))):
    jr_skill, jr_count = most_common_jr_skills[i]
    sr_skill, sr_count = most_common_sr_skills[i]
    print(f"{i+1:<5} {jr_skill:<20} {jr_count:<7} {sr_skill:<20} {sr_count:<7}")

## For the most part, these lists are in agreement. I'm honestly a little surprised not to see more differentiation.
## I think what would be interesting is to see which skills are associated with higher salaries within each experience level.
## Let's start with junior roles.
jr_skill_salary = {}
for skills, salary in zip(df_filtered[df_filtered['Seniority_Binary'] == 'junior']['skills'].dropna(), df_filtered[df_filtered['Seniority_Binary'] == 'junior']['Cleaned_Salary'].dropna()):
    skill_list = [skill.strip(" []'\"").lower() for skill in skills.split(',') if skill.strip(" []'\"")]
    for skill in skill_list:
        if skill not in jr_skill_salary:
            jr_skill_salary[skill] = []
        jr_skill_salary[skill].append(salary)

## What that did was create a dictionary where each skill maps to a list of salaries for junior roles requiring that skill.
## Now let's get some summary stats for each skill.
jr_skill_salary_summary = {skill: (np.mean(salaries), np.median(salaries), np.std(salaries), len(salaries)) for skill, salaries in jr_skill_salary.items() if len(salaries) >= 2}
jr_skill_salary_summary_sorted = sorted(jr_skill_salary_summary.items(), key=lambda x: x[1][0], reverse=True)
## Now we have a dictionary containing skills and summary stats, sorted by mean salary.
## Let's print out a table with the top 25 skills and their associated salary stats.
print(f"{'No.':<5} {'Skill':<20} {'Mean Salary':<15} {'Median Salary':<15} {'Std Dev':<10} {'Count':<7}")
print("-" * 80)
for i, (skill, (mean_salary, median_salary, std_dev, count)) in enumerate(jr_skill_salary_summary_sorted[:25]):
    print(f"{i+1:<5} {skill:<20} {mean_salary:<15.2f} {median_salary:<15.2f} {std_dev:<10.2f} {count:<7}")
    
## One thing I'm noticing is that these standard deviations are very high.
## What this tells me is that the roles I've broadly classified as "junior" probably represent a wide range of actual experience levels.
## My theory at this point would be that the top skills are really just the "tools of the trade" and that, for these junior roles at least,
## there's unlikely to be a strong correlation between specific skills and salary. Also, pay for some of these jobs is shockingly low.
## At some point later I might want to see if the covariance of skill pairs for some of these tops skills is better correlated with salary.
## For now, I'll take a look at senior roles next and see if there's more differentiation there.
sr_skill_salary = {}
for skills, salary in zip(df_filtered[df_filtered['Seniority_Binary'] == 'senior']['skills'].dropna(), df_filtered[df_filtered['Seniority_Binary'] == 'senior']['Cleaned_Salary'].dropna()):
    skill_list = [skill.strip(" []'\"").lower() for skill in skills.split(',') if skill.strip(" []'\"")]
    for skill in skill_list:
        if skill not in sr_skill_salary:
            sr_skill_salary[skill] = []
        sr_skill_salary[skill].append(salary)


sr_skill_salary_summary = {skill: (np.mean(salaries), np.median(salaries), np.std(salaries), len(salaries)) for skill, salaries in sr_skill_salary.items() if len(salaries) >= 2}
sr_skill_salary_summary_sorted = sorted(sr_skill_salary_summary.items(), key=lambda x: x[1][0], reverse=True)

print(f"{'No.':<5} {'Skill':<20} {'Mean Salary':<15} {'Median Salary':<15} {'Std Dev':<10} {'Count':<7}")
print("-" * 80)
for i, (skill, (mean_salary, median_salary, std_dev, count)) in enumerate(sr_skill_salary_summary_sorted[:25]):
    print(f"{i+1:<5} {skill:<20} {mean_salary:<15.2f} {median_salary:<15.2f} {std_dev:<10.2f} {count:<7}")

## Ok, now we're getting somwhere. Scala is obviously showing itself as a high-value skill. My bet is that this has to do with who is hiring.
## That is to say, it's probably big companies who are utilizing Scala for large dataset processing, automations, etc.
## I'll have to do more reasearch on that (and see if I can find a way to work with Scala for a project in my portfolio),
## but the language itself doesn't really look to be that different from Python or R for data science tasks. If a company has already sunk
## significant resources into Scala, though, that would explain why they are willing to pay a premium for it.
## Same is likely the case for Spark and some the other big data tools we're seeing here.
## Interestingly, bash shows up in the senior skills, but not the junior skills. I would guess this is because senior roles are expected to own
## more of the end-to-end data pipeline, which often involves some bash scripting for OS-level automations.
## I'm also somewhat surprised not to see more generic skills (communication, teamwork, agile, MS Office, etc.).
## Maybe these things aren't as necessary for these roles, as they might not be client-facing? or maybe they just assume everyone has them?


## Let's look at job titles next.
title_list = df['job_title'].dropna().unique()
title_counts = df['job_title'].dropna().value_counts()

for title in title_list:
        plt.figure(figsize=(8, 5))
        df_filtered[df_filtered['job_title']==title]['Cleaned_Salary'].plot.hist(bins=10, edgecolor = 'black')
        plt.xlabel('Salary')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of {title} Salaries")
        plt.grid(True)
        plt.show(block = False)

input('Press Enter to continue...')