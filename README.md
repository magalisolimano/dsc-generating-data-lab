# Generating Data - Lab

## Introduction

In this lab, we shall practice some of the data generation techniques that we saw in the previous lesson in order to generate datasets for regression and classification purposes. We will run a couple of simple simulations to help us generate different datasets by controlling noise and variance parameters in the data generation process. We will also look at the statistical indicators and visual output to see how these parameters affect the accuracy of an algorithm. 

## Objectives
In this lab you will:

- Generate datasets for classification problems
- Generate datasets for regression problems

## Generate Data for Classfication

Use `make_blobs()` to create a binary classification dataset with 100 samples, 2 features, and 2 centers (where each center corresponds to a different class label). Set `random_state = 42` for reproducibility.

_Hint: Here's a link to the documentation for_ [`make_blobs()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).


```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
```

Place the data in a `pandas` DataFrame called `df`, and inspect the first five rows of the data. 

_Hint: Your dataframe should have three columns in total, two for the features and one for the class label._ 


```python
import pandas as pd
df = pd.DataFrame(X, columns=["X1", "X2"])
df["y"] = y
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.988372</td>
      <td>8.828627</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.722930</td>
      <td>3.026972</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.053580</td>
      <td>9.125209</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.461939</td>
      <td>3.869963</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.867339</td>
      <td>3.280312</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-1.478198</td>
      <td>9.945566</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>-1.593796</td>
      <td>9.343037</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>3.793085</td>
      <td>0.458322</td>
      <td>1</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-2.728870</td>
      <td>9.371399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-2.504084</td>
      <td>8.779699</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>



Create a scatter plot of the data, while color-coding the different classes.


```python
# Matplotlib approach
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sc = ax.scatter(df["X1"], df["X2"], c=df["y"], cmap="viridis")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend(*sc.legend_elements());
```


    
![png](index_files/index_7_0.png)
    



```python
# pandas plotting approach
df.plot.scatter(x="X1", y="X2", c="y", colormap="viridis", sharex=False);
```


    
![png](index_files/index_8_0.png)
    


Repeat this exercise two times by setting `cluster_std = 0.5` and `cluster_std = 2`. 

Keep all other parameters passed to `make_blobs()` equal. 

That is:
* Create a classification dataset with 100 samples, 2 features, and 2 centers using `make_blobs()` 
    * Set `random_state = 42` for reproducibility, and pass the appropriate value for `cluster_std`  
* Place the data in a `pandas` DataFrame called `df`  
* Plot the values on a scatter plot, while color-coding the different classes 

What is the effect of changing `cluster_std` based on your plots? 


```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=0.5, random_state=42)
df = pd.DataFrame(X, columns=["X1", "X2"])
df["y"] = y
df.plot.scatter(x="X1", y="X2", c="y", colormap="viridis", sharex=False);
```


    
![png](index_files/index_10_0.png)
    



```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=2, random_state=42)
df = pd.DataFrame(X, columns=["X1", "X2"])
df["y"] = y
df.plot.scatter(x="X1", y="X2", c="y", colormap="viridis", sharex=False);
```


    
![png](index_files/index_11_0.png)
    



```python

# When setting cluster_std = 0.5 and keeping all other parameters the same, 
# we obtain two different clusters centered at the same positions as beforehand, 
# but the points in each cluster are closer to the centers of the clusters. 
# 
# When setting cluster_std = 2 and keeping all other parameters equal, 
# we obtain two different clusters centers at the same positions as beforehand,
# but more spread out about the centers of the clusters. 
# 
# cluster_std controls the spread of the data about the center of the clusters 
# we've created. 
```

## Generate Data for Regression

Create a function `reg_simulation()` to run a regression simulation creating a number of datasets with the `make_regression()` data generation function. Perform the following tasks:

* Create `reg_simulation()` with `n` (noise) and `random_state` as input parameters
    * Make a regression dataset (X, y) with 100 samples using a given noise value and random state
    * Plot the data as a scatter plot 
    * Calculate and plot a regression line on the plot and calculate $R^2$ (you can do this in `statsmodels` or `sklearn`)
    * Label the plot with the noise value and the calculated $R^2$ 
    
* Pass a fixed random state and values from `[10, 25, 40, 50, 100, 200]` as noise values iteratively to the function above 
* Inspect and comment on the output 


```python

# Import necessary libraries
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

def reg_simulation(n, random_state):
    # Generate X and y
    X, y = make_regression(n_samples=100, n_features=1, noise=n, random_state=random_state)
    # Fit a linear regression model to X, y
    reg = LinearRegression().fit(X, y)
    
    # Use X,y to draw a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', s=10, label='Data')
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    
    # Label and plot the regression line
    ax.plot(X, reg.predict(X), color='black', label='Model')
    fig.suptitle(f'Noise: {n}, R-Squared: {round(reg.score(X,y), 2)}')
    ax.legend()

random_state = 42

for n in [10, 25, 40, 50, 100, 200]:
    reg_simulation(n, random_state)
```


    
![png](index_files/index_14_0.png)
    



    
![png](index_files/index_14_1.png)
    



    
![png](index_files/index_14_2.png)
    



    
![png](index_files/index_14_3.png)
    



    
![png](index_files/index_14_4.png)
    



    
![png](index_files/index_14_5.png)
    



```python

# As the noise level increases, the data points get farther away from
# the model line and the coefficient of determination of our model fit decreases. 
```

## Summary 

In this lesson, we learned how to generate random datasets for classification and regression problems. We ran simulations for this and fitted simple models to view the effect of random data parameters including noise level and standard deviation on the performance of parameters, visually as well as objectively. These skills will come in handy while testing model performance and robustness in the future. 
