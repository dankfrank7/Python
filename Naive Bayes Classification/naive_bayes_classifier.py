import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, f1_score
import naive_bayes_fns as naive_bayes 

## 1 Prepare data
# Load in data
data = pd.read_csv('Breast_cancer_data.csv')

# Check that variables are independent
corr = data.iloc[:,:-1].corr(method='pearson')
    #                  mean_radius  mean_texture  mean_perimeter  mean_area  mean_smoothness
    # mean_radius         1.000000      0.323782        0.997855   0.987357         0.170581
    # mean_texture        0.323782      1.000000        0.329533   0.321086        -0.023389
    # mean_perimeter      0.997855      0.329533        1.000000   0.986507         0.207278
    # mean_area           0.987357      0.321086        0.986507   1.000000         0.177028
    # mean_smoothness     0.170581     -0.023389        0.207278   0.177028         1.000000
cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)

# Remove highly correlated data
data = data[['mean_radius', 'mean_texture', 'mean_smoothness', 'diagnosis']]

# Plot thie histograms of each feature
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
colors = cmap(np.linspace(0, 1, data.shape[1]-1))
for i, ax in enumerate(axes):
    sns.histplot(data, ax=ax, x=data.iloc[:,i], kde=True, color=colors[i])

plt.show(block=False)

## Test Models
# Test Gaussian model
train, test = train_test_split(data, test_size=.2, random_state=41)

X_test = test.iloc[:,:-1].values 
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes.naive_bayes_gaussian(train, X=X_test, Y='diagnosis')

print('Gaussian Model')
print(confusion_matrix(Y_test, Y_pred))
print(f'Accuracy: {round(f1_score(Y_test, Y_pred)*100,2)}%')

# Convert continuous features to categorical features 
data_cat = data.copy()
for col in data.columns[:-1]: 
    new_col_name = f'cat_{col}'
    data_cat[new_col_name] = pd.cut(data[col], bins=3, labels=[0,1,2])
    data_cat.drop(columns=col, inplace=True)

# Move target column to the final column
col_names = list(data_cat.columns)
new_column_order = col_names[1:] + col_names[:1]
data_cat = data_cat[new_column_order]

# Test categorical model
train_cat, test_cat = train_test_split(data_cat, test_size=.2, random_state=41)
X_test_cat = test_cat.iloc[:,:-1].values 

Y_pred_cat = naive_bayes.naive_bayes_categorical(train_cat, X=X_test_cat, Y='diagnosis')

print('Categorical Model')
print(confusion_matrix(Y_test, Y_pred_cat)) 
print(f'Accuracy: {round(f1_score(Y_test, Y_pred_cat)*100,2)}%')
    

input('Press enter to close figures..')