import pandas as pd
import seaborn as sns

ad_data = pd.read_csv('./data/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

ad_data['Age'].hist(bins=40)

sns.set_style('whitegrid')
sns.jointplot(x='Age', y='Area Income', data=ad_data)

sns.jointplot(
    x='Age',
    y='Daily Time Spent on Site',
    data=ad_data,
    color='red',
    kind='kde'
)

sns.jointplot(
    x='Daily Internet Usage',
    y='Daily Time Spent on Site',
    data=ad_data,
    color='green'
)

sns.pairplot(data=ad_data, hue='Clicked on Ad')
