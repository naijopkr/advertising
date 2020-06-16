import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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

sns.heatmap(
    data=ad_data.isnull(),
    yticklabels=False,
    cbar=False,
    cmap='viridis'
)

predictors = [
    'Age',
    'Area Income',
    'Daily Internet Usage',
    'Daily Time Spent on Site'
]
X = ad_data[predictors]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=101
)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

y_pred = logmodel.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
