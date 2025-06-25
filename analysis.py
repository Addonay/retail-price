# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Data Loading

# %%
# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("data/dataset.csv")
df

# %%
df.head()

# %% [markdown]
# Data  Cleaning

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %%
df.describe().transpose()

# %%
df.nunique()

# %%
df.info() 

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(x='product_category_name', y='total_price', data=df)
plt.title('Product Category vs. Total Price')
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(x='product_category_name', y='unit_price', data=df)
plt.title('Product Category vs. Unit Price')
plt.xticks(rotation=90)
plt.show()
