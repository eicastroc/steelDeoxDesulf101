
```{code-cell}

```

```{code-cell}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r


# # Metallurgical context

# Extreme value statistics is a branch of statistics dealing with assesing the probability of occurence of an event more extreme than any previously observed one.
# 
# In a metallurgical quality and materials characterization context, an extreme event would be the probability that the largest size of a microstructural feature (e.g., inclusions, graphite flakes, precipitates, porosities, cavities, etc.) in a product be larger than a given threshold value. In this context, the (Gumbel) Extreme Value distribution is used to estimate these probabilities.
# 
# The purpose of this Notebook is thus to provide a practical-aid using Python libraries, for performing extreme value analysis of microstructural features in materials. Two topics are discussed: 
# 
# 1) Estimation of (Gumbel) Extreme Value distribution parameters.
# 
# 2) Construction of Gumbel plots.
# 
# A **References / Further Reading** section is provided at the end of this notebook.

# # Fundamentals for Extreme Value Analysis

# In this section, the empirical cumulative density function is first discussed. Then the (Gumbel) Extreme Value Distribution is presented, along with two methodologies for estimating the distribution parameters (moments method and maximum likelihood method). Finally, the procedure for the construction of a Gumbel plot is presented.

# ## Empirical Cumulative Density Function

# Each of the $N$ measurements of size of a microstructural feature can be represented as $x_i$, where $1 \le i \le N$.
# 
# To estimate the __Empirical Cumulative Density Function (ECDF)__ the data of measurements of size of a microstructural feature, $x_i$,  are ordered in ascending order so that:
# 
# $$x_1 \le x_2 \le x_3 \ldots \le x_N$$
# 
# Then the ECDF is calculated with the following equation:
# 
# $$P_i = \frac{i}{N+1}$$
# 
# The function `eCDF()`, presents an implementation of the ECDF:
```

```{code-cell}


def eCDF(df:str | pd.DataFrame):
    """ 
    empirical cumulative distribution function for inclusion size,
    or any other microstructural feature that can be analyzed with
    Extreme Value methodology.

    Parameters
    ----------
    df: pandas.DataFrame
        data of measurements with three columns: [run, specimen, Y].
        The column names are in concordance with ASTM E2283 nomenclature.
            - run: a string identifying each of the four runs: [A, B, C, D].
            - specimen: An integer identifying a set of measurements over a
                          defined surface area (for inclusions, 150 microns).
                          For each run, 6 specimens are required: [1, 2, 3, 5, 6].
            - Y: Measurements of the microstructural feature that will be studied 
                    using Extreme Value Methodology. In the case of inclusions, 
                    Y correponds to the longest inclusion length in an specimen.
    
    Returns
    -------
    Ysorted: np.ndarray
        microstructural feature measurements, in ascending order.
    ecdf: numpy.ndarray
        empirical cumulative distribution function values.
    """
    Ysorted = df.Y.sort_values().reset_index(drop=True).to_numpy()  # sort values
    Nvalues = len(Ysorted)
    ecdf = np.linspace(1, Nvalues, Nvalues) / (Nvalues+1)    # estimate ecdf
    return Ysorted, ecdf


# ## (Gumbel) Extreme Value Distribution

# ### Probability Density Function and Cumulative Density Functions

# 
# 
# The __Probability Density Function (PDF)__ for the two-parameter (Gumbel) Extreme Value Distribution is given by:
# 
# $$f(x) = \frac{1}{\delta} \left[\exp\left(-\frac{x-\lambda}{\delta}\right)\right] \times \exp\left[-\exp\left(-\frac{x-\lambda}{\delta}\right)\right]$$
# 
# The __Cumulative Density Function (CDF)__ is given by:
# 
# $$F(x) = \exp\left(-\exp\left(-\frac{x - \lambda}{\delta}\right)\right)$$
# 
# 
# where:
# 
# - $x$: size of the largest microstructural feature in each control area, $A_0$.
# 
# - $\lambda$: location parameter of the (Gumbel) Extreme Value distribution function.
# 
# - $\delta$: scale parameter of the (Gumbel) Extreme Value distribution function

# ### Estimation of distribution parameters (Moments Method)

# The parameters of the (Gumbel) Extreme Value distribution can be estimated with the following equations:
# 
# $$\delta_{mom} = \frac{s \, \sqrt{6}}{\pi}$$
# 
# $$\lambda_{mom} = \bar{X} - 0.5772 \, \delta_{mom}$$
# 
# The subscript, $_{mom}$ , indicates that the estimates are obtained by the moments method, in which:
# 
# - The first moment is the mean largest size of the dataset of measurements of size of the microstructural feature, $\bar{X}$. It is estimated as:
# 
# $$\bar{X} = \frac{1}{N} \sum_{i=1}^{n} x_i$$
# 
# - The square root of the second moment is the standard deviation of the dataset of measurements of size of microstructural feature, $s$. It is estimated as:
# 
# $$s = \left[\sum_{i=1}^{N} \frac{\left(x_i - \bar{X}\right)^2}{N-1}\right]^{0.5}$$
# 
# 
# The function `fitEVmom()`, presents an implementation of the procedure for estimating the Extreme Value distribution parameters with the Moments Method:
```

```{code-cell}


def fitEVmom(df:str | pd.DataFrame, verbose:bool = True):
    """
    Fit (Gumbel) Extreme Value distribution to measurements
    of a microstructural feature (e.g., inclusion size) using
    Moments Method (as presented in ASTM E2283 norm)

    Parameters
    ----------
    df: pandas.DataFrame
        data of measurements with three columns: [run, specimen, Y].
        The column names are in concordance with ASTM E2283 nomenclature.
            - run: a string identifying each of the four runs: [A, B, C, D].
            - specimen: An integer identifying a set of measurements over a
                          defined control area.
                          For each run, 6 specimens are required: [1, 2, 3, 5, 6].
            - Y: Measurements of the microstructural feature that will be studied 
                    using Extreme Value Methodology. In the case of inclusions, 
                    Y correponds to the longest inclusion length in an specimen.
    
    Returns
    -------
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distriution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distriution.
    """
    Ymean = df.Y.mean() # mean
    Ystd = df.Y.std()   # std. dev. pop.
    delta = Ystd * np.sqrt(6) / np.pi   # scale parameter
    lamb = Ymean - 0.5772 * delta       # location parameter
    if verbose==True:
        print('Gumbel distribution params.')
        print('lambda: {:.4f}'.format(lamb))
        print('delta : {:.4f}'.format(delta))
        print('*MoM est.')
    return lamb, delta    


# ### Estimation of distribution parameters (Maximum Likelihood Method)

# The method of maximum likelihood is based on the approach that the best estimation of distribution parameters are those that maximize the likelihood of obtaining the set of measurements of the given microstructural feature. This is done using a numerical method sought for determining the values of the distribution parameters which maximize the sum of the logarithms of the probability density function, when evaluating the data set. When the natural logarithm is chosen, the optimization function is of the form:
# 
# $$LL=\sum_{i=1}^{N} \ln \left(f\left(x_i, \lambda, \delta \right) \right)$$
# 
# where the optimization procedure consists on finding the $\lambda_{ML}$ and $\delta_{ML}$ that maximize the value of $LL$. The subscript, $_{ML}$, indicate that the estimates are obtained by the Maximum Likelihood method.
# 
# The function `fitEVml()`, presents an implementation of the procedure for estimating the Extreme Value distribution parameters with the Maximum Likelihood Method. This implementation is simply a wrapper of one of the methods of the `gumbel_r` class from the `scipy.stats` library:
```

```{code-cell}


def fitEVml(df:str | pd.DataFrame, verbose:bool = True):
    """
    Fit (Gumbel) Extreme Value distribution to measurements
    of a microstructural feature (e.g., inclusion size) using
    Maximum Likelihood methods (as implemented in scipy.stats)
    
    Parameters
    ----------
    df: pandas.DataFrame
        data of measurements with three columns: [run, specimen, Y].
        The column names are in concordance with ASTM E2283 nomenclature.
            - run: a string identifying each of the four runs: [A, B, C, D].
            - specimen: An integer identifying a set of measurements over a
                          defined control area.
                          For each run, 6 specimens are required: [1, 2, 3, 5, 6].
            - Y: Measurements of the microstructural feature that will be studied 
                    using Extreme Value Methodology. In the case of inclusions, 
                    Y correponds to the longest inclusion length in an specimen.
    
    Returns
    -------
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distriution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distriution.
    """    
    evDist = gumbel_r.fit(df.Y) #ML fitting
    delta = evDist[1]   # scale parameter
    lamb = evDist[0]    # location parameter
    if verbose==True:
        print('Gumbel distribution params.')
        print('lambda: {:.4f}'.format(lamb))
        print('delta : {:.4f}'.format(delta))
        print('*ML est.')
    return lamb, delta


# ### Confidence interval of estimations of sizes of microstructural features

# The standard error of any microstructural feature of size $x$, based on the maximum likelihood method is given by:
# 
# $$\mathrm{SE}(x) = \delta \sqrt{\frac{1.109+0.514\,y+0.608\,y^2}{N}}$$
# 
# The 95% confidence interval is given by:
# 
# $$95\%\,\mathrm{CI} = \pm 2\, \mathrm{SE}(x)$$
# 
# where:
# 
# - $x$: size of the microstructural feature.
# 
# - $\lambda$: location parameter of the (Gumbel) Extreme Value distribution function.
# 
# - $\delta$: scale parameter of the (Gumbel) Extreme Value distribution function.
# 
# - $y$: reduced variable, defined as: $y=\frac{x - \lambda}{\delta}$.
# 
# - $N$: number of measurements used when performing the fitting of the (Gumbel) Extreme Value distribution.
# 
# 
```

```{code-cell}


def calcSE(x:float, lamb:float, delta:float, N:int):
    """
    Estimate the Standard Error (SE) of size estimation for
    a population of microstructural features that has been fit
    to the (Gumbel) Extreme Value distribution.
    
    Parameters
    ----------
    x: Float
        size of microstructural feature
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distribution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distribution.
    N: Int
        number of measurements
    
    Returns
    -------
    SE: Float
        standard error of size estimations
    """    
    y = (x - lamb) / delta
    SE = delta * np.sqrt((1.109 + 0.514*y + 0.608*y**2)/N)
    return SE


# ## Gumbel plot

# The Gumbel plot is a visualization tool developped in the times when computers where not readily available for the general public. Originally, these kind of plots were made by hand-plotting the linearized (Gumbel) Extreme Value Cumulative Distribution Function and the data-points on 2-cycle log-probability paper.

# ### Linearization of the Cumulative Density Function:

# First, the __reduced variable__, $y$, is defined as:
# 
# $$y = \frac{x - \lambda}{\delta}$$
# 
# The Cumulative Density Function of the (Gumbel) Extreme Value distribution, $F(x)$, can be then rewriten as a function of $y$:
# 
# $$F(x) = \exp\left(-\exp\left(-\frac{x - \lambda}{\delta}\right)\right)$$
# 
# $$F(y) = \exp\left(-\exp\left(-y\right)\right)$$
# 
# The linearized cumulative distribution function is then written as:
# 
# $$-\ln\left(-\ln\left(F(x)\right)\right) = \frac{x - \lambda}{\delta}$$
# 
# $$-\ln\left(-\ln\left(F(y)\right)\right) = y$$
# 
# where $F(x) = F(y) = P$, is the probability thet the largest microstructural feature be as large as size $x$.
# 
# The function `calcRedVar()`, presents an implementation of the procedure for linealizing the Cumulative Density Function of the (Gumbel) Extreme Value distribution:
```

```{code-cell}


def calcRedVar(F:float):
    """ 
    reduced variable, y, used in the procedure of linearization 
    of the (Gumbel) Extreme Value CDF.
    
    Parameters
    ----------
    F: float
        Cumulative density function F(y)
    
    Returns
    -------
    y: float
        linealized (Gumbel) Extreme Value cumulative density function.
    """    
    y = -1.0 * np.log(-np.log(F))
    return y


# Once the best estimates for the location $(\lambda)$ and scale $(\delta)$ parameters are determined, the expected maximum size of the microstructural feature can be estimated for a given probability using the equation:
# 
# $$x = \delta y + \lambda$$
# 
# or
# 
# $$x = \delta \left[-\ln\left(-\ln\left(P\right)\right)\right] + \lambda$$
# 
# This corresponds to the equation of a straight line, where the slope is given by the scale parameter $(\delta)$, and the intercept is given by the location parameter $(\lambda)$.

# ### Construction of the Gumbel plot

# The development of the Gumbel plot preceeds the advent of computer methods. As such, these kind of plots were useful to enable the estimation of (Gumbel) Extreme Value distribution parameters by graphical methods. Today, they serve as a tool for verifying the goodness-of-fit of the distribution parameters against the set of data.
# 
# The steps for constructing one of such plots are as follows:

# - Perform work on data set using the Empirical Cumulative distribution function:
#     - Sort the data of measurements of size of microstructural feature, $x$, in ascending order.
#     - Estimate the empirical cumulative distribution function (ECDF) of the data, $P$.
#     - Estimate the value of the reduced variable, for the ECDF calculated in the previous step, $y=-\ln\left(-\ln\left(P\right)\right)$.

# - Fit the (Gumbel) Extreme value distribution to the data set using a numerical method:
#     - Use either Moments Method or Maximum Likelihood method for estimation of the distribution parameters, $\delta$ and $\lambda$.
#     - Obtain the estimates of size of microstructural feature, $x_{ML}$, for the given empirical probabilities of the data, $P$.

# - Plot the data
#     - Plot the linealized empirical distribution function vs the sorted size measurements, $-\ln\left(-\ln\left(P\right)\right)$ vs $x$.
#     - Plot the linealized empirical distribution function vs the Extreme Value size estimations, $-\ln\left(-\ln\left(P\right)\right)$ vs $x_{ML}$.
#     - Plot the lines representing the confidence interval of the Extreme Value estimations.

# The function `gumbelPlot()`, presents an implementation of the procedure for constructing a Gumbel plot:
```

```{code-cell}


def gumbelPlot(df:str | pd.DataFrame, gamma:float = None, delta:float = None, ax=None):

    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    #### Perform work on data set using the ECDF
    xSorted, ecdf = eCDF(df)                 # ecdf
    redVar = -1.0 * np.log(-np.log(ecdf))    # reduced variable

    #### Fit the distribution to the data set
    if ((gamma==None) or (delta==None)):
        lamb, delta = fitEVml(df)           # Maximum Likelihood
    xML = delta*redVar + lamb               # x-value estimation
    
    #### plot data
    ax.plot(xSorted, redVar, ls='', marker='o', color='k', markerfacecolor='w')    # ECDF
    ax.plot(xML, redVar, ls='-', color='k')                                        # EV-Distribution
    
    #### plot confidence interval
    N = len(df)
    SE = calcSE(xML, lamb, delta, N)
    xmin = xML - 2.0*SE
    xmax = xML + 2.0*SE
    ax.plot(xmin, redVar, ls='--', color='k')                          # EV-Distribution min 95%CI
    ax.plot(xmax, redVar, ls='--', color='k')                          # EV-Distribution max 95%CI

    #### plot layout
    ax.set_ylabel('Reduced variable')
    ax.set_xlabel('x (size, length, etc.)')
    ax.grid(ls='--', color='lightgray')
    ax.annotate("Gumbel params.", fontweight='bold',
                xy=(0.15, 0.85), xycoords='subfigure fraction')
    ax.annotate(r"$\lambda =${:.4f}".format(lamb),
                xy=(0.15, 0.80), xycoords='subfigure fraction')
    ax.annotate(r"$\delta =${:.4f}".format(delta),
                xy=(0.15, 0.75), xycoords='subfigure fraction')


# # Example of application of Extreme Value Analysis

# The dataset for this example is taken from the ASTM E-2283 standard. The dataset consists of 24 measurements of the largest inclusion found in six steel specimens(specimen: 1 - 6), with the procedure being repeated four times (run: A - D).
# 
# Below, the data is put into Pandas DataFrame format, and the function `gumbelPlot()` defined above is used to fit a (Gumbel) Extreme Value Distribution function to the data, and draw a Gumbel plot for the data.
```

```{code-cell}


# Dataset from ASTM-E2283
specimen = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
run = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D']
Y = [40.29, 37.24, 29.03, 52.46, 62.21, 33.98, 30.73, 37.43, 35.00, 44.82, 66.13, 48.55, 73.48, 44.79, 70.87, 59.83, 22.18, 64.32, 78.91, 46.53, 94.28, 49.15, 82.39, 37.43]
```

```{code-cell}


# Construct dataframe
df = pd.DataFrame(list(zip(run, specimen, Y)),
                     columns=['run', 'specimen', 'Y'])
```

```{code-cell}


# Show the dataframe
#df.sort_values(by='Y', ascending=True, inplace=True) # uncomment for sorting by size
display(df)
```

```{code-cell}


# Construct the Gumbel plot (with estimation of parameters)
gumbelPlot(df)


# # References / Further reading

# - ASTM Norm: [ASTM E2283: Standard Practice for Extreme Value Analysis of Nonmetallic Inclusions in Steel and Other Microstructural Features](https://www.astm.org/e2283-08r19.html)
# 
# - Book: [Y. Murakami, Metal Fatigue (2002)](https://www.sciencedirect.com/book/9780080440644/metal-fatigue)
# 
# - Wikipedia site: [Gumbel Distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)
# 
# - Scipy documentation: [scipy.stats.gumbel_r](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
```
