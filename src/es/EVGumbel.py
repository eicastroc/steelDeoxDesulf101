#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r


# # Contexto metalúrgico

# La estadística de valores extremos es una rama de la estadística que se interesa en estimar la probabilidad de que ocurra un evento más extremo que cualquier evento observado previamente.
# 
# Dentro del contexto de calidad metalúrgica y de caracterización de materiales, un evento extremo podría ser la probabilidad de que una característica microstructural en un material (e.g., inclusiones, hojuelas de grafito, precipitados, porosidades, cavidades, etc.) sea más grande que a un tamaño crítico. En este contexto, la Distribución (Gumbel) de Valores Extremos se utiliza para estimar las probabilidades.
# 
# El objetivo de este Notebook es proveer una guía práctica para realizar análisis de características microestructurales de materiales, utilizando librerías de Python. Se presentan dos temas:
# 
# 1) Estimación de parámetros de la Distribución (Gumbel) de Valores Extremos.
# 
# 2) Construcción de gráficos de Gumbel.
# 
# 
# Al final de este Notebook, se provee una sección con  **Referencias**.

# # Fundamentos de Análisis de Valores Extremos

# En esta sección se presenta primeramente la Distribución Empírica (de Probabilidad Acumulada). Después se presenta la Distribución (Gumbel) de Valores Extremos, junto con dos metodologías para la estimación de los parámetros de la distribución (método de momentos, y método de la máxima verosimilitud). Finalmente, se presenta el procedimiento para la construcción de un gráfico de Gumbel. 

# ## Distribución Empírica

# Cada una de las $N$ mediciones de tamaño de una característica microestructural se puede representar como $x_i$, en donde $1 \le i \le N$.
# 
# Para estimar la __Distribución Empírica__, los datos de tamaño de una característica microestructural, $x_i$, se ordenan en orden ascendente de tal forma que:
# 
# $$x_1 \le x_2 \le x_3 \ldots \le x_N$$
# 
# La distribución empírica se calcula como:
# 
# $$P_i = \frac{i}{N+1}$$
# 
# La función `eCDF()`, presenta una implementación de la Distribución empírica:

# In[2]:


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


# ## Distribución (Gumbel) de Valores Extremos

# ### Funciones de densidad de probabilidad (PDF) y densidad acumulada (CDF)

# 
# La función de densidad de probabilidad (PDF) para la Distribución (Gumbel) de Valores Extremos está dada por:
# 
# $$f(x) = \frac{1}{\delta} \left[\exp\left(-\frac{x-\lambda}{\delta}\right)\right] \times \exp\left[-\exp\left(-\frac{x-\lambda}{\delta}\right)\right]$$
# 
# La función de densidad acumulada (CDF) está dada por:
# 
# $$F(x) = \exp\left(-\exp\left(-\frac{x - \lambda}{\delta}\right)\right)$$
# 
# 
# donde:
# 
# - $x$: tamaño de la característica microestructural más grande en un área de control, $A_0$.
# 
# - $\lambda$: parámetro de locación de la Distribución (Gumbel) de Valores Extremos.
# 
# - $\delta$: parámetro de escala de la Distribución (Gumbel) de Valores Extremos.

# ### Estimación de parámetros de la distribución (Método de momentos)

# Los parámetros de la Distribución (Gumbel) de Valores Extremos se estiman mediante las siguientes ecuaciones:
# 
# $$\delta_{mom} = \frac{s \, \sqrt{6}}{\pi}$$
# 
# $$\lambda_{mom} = \bar{X} - 0.5772 \, \delta_{mom}$$
# 
# El subindice, $_{mom}$ , indica que los estimados se obtienen por el método de momentos, en el cuál:
# 
# - El primer momento es el promedio del tamaño más grande de la característica microestructural, $\bar{X}$, calculado a partir de la serie de datos:
# 
# $$\bar{X} = \frac{1}{N} \sum_{i=1}^{n} x_i$$
# 
# - La raíz cuadrada del segundo momento es la desviación estandar de la serie de datos:
# 
# $$s = \left[\sum_{i=1}^{N} \frac{\left(x_i - \bar{X}\right)^2}{N-1}\right]^{0.5}$$
# 
# 
# La función `fitEVmom()` es una implementación del procedimiento para estimar los parámetros de la Distribución (Gumbel) de Valores Extremos mediante el método de momentos.

# In[3]:


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


# ### Estimación de parámetros de la distribución (Método de máxima verosimilitud)

# El método de máxima verosimilitud se basa en el postulado que el mejor estimado de parámetros para la distribución es aquel que maximice la verosimilitud entre la distribución y los datos de tamaño de característica microestructural. Este procedimiento se lleva a cabo mediante un método numérico en el cuál se buscan los parámetros de distribución que maximicen una función objetivo dada por la suma de los logaritmos de la función de densidad de probabilidad evaluada para todos los datos. Cuando se utiliza logaritmo natural, la función de optimización es de la forma:
# 
# $$LL=\sum_{i=1}^{N} \ln \left(f\left(x_i, \lambda, \delta \right) \right)$$
# 
# para la cual, el procedimiento de optimización consiste en encontrar los valores de $\lambda_{ML}$ y $\delta_{ML}$ que maximicen el valor de $LL$. El subídice $_{ML}$, indica que los estimados son obtenidos mediante el método de máxima verosimilitud.
# 
# 
# La función `fitEVml()`, es un wrapper para la clase `gumbel_r` de la librería `scipy.stats`, que permite estimar los parámetros mediante el método de máxima verosimilitud.

# In[4]:


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


# ### Intervalo de confianza para la estimaciones de tamaño de características microestructurales

# El error estándar para el tamaño de una característica microestructural, $x$, basado en los résultados del método de máxima verosimilitud, está dado por:
# 
# 
# $$\mathrm{SE}(x) = \delta \sqrt{\frac{1.109+0.514\,y+0.608\,y^2}{N}}$$
# 
# El intervalor de confianza de 95% está dado por:
# 
# $$95\%\,\mathrm{CI} = \pm 2\, \mathrm{SE}(x)$$
# 
# donde:
# 
# - $x$: tamaño de la característica microestructural.
# 
# - $\lambda$: parámetro de locación de la Distribución (Gumbel) de Valores Extremos.
# 
# - $\delta$: parámetro de escala de la Distribución (Gumbel) de Valores Extremos.
# 
# - $y$: variable reducida, definida como: $y=\frac{x - \lambda}{\delta}$.
# 
# - $N$: numero de datos de mediciones utilizadas en el proceso de ajuste de la Distribución (Gumbel) de Valores Extremos.

# In[5]:


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


# ## Gráfico de Gumbel

# El gráfico de Gumbel es una herramienta de visualización desarrollada en tiempos en los que no había acceso a computadoras para el público en general. Originalmente, este tipo de gráficos se hacían trazando a mano la función de probabilidad acumulada de la Distribución (Gumbel) de Valores Extremos linealizada y los puntos de datos en papel logarítmico de dos ciclos.

# ### Linealización de la función de probabilidad acumulada:

# Primeramente, se define a la __variable reducida__, $y$, como:
# 
# $$y = \frac{x - \lambda}{\delta}$$
# 
# Entonces, la función de probabilidad acumulada de la Distribución (Gumbel) de Valores Extremos, $F(x)$, se puede reescribir en función de $y$:
# 
# $$F(x) = \exp\left(-\exp\left(-\frac{x - \lambda}{\delta}\right)\right)$$
# 
# $$F(y) = \exp\left(-\exp\left(-y\right)\right)$$
# 
# La función linealizada se puede reescribir posteriormente como:
# 
# $$-\ln\left(-\ln\left(F(x)\right)\right) = \frac{x - \lambda}{\delta}$$
# 
# $$-\ln\left(-\ln\left(F(y)\right)\right) = y$$
# 
# en donde $F(x) = F(y) = P$, es la probabilidad de que la característica microestructural más grande sea tan grande como el tamaño $x$.
# 
# La función `calcRedVar()`, es una implementación del proceso para linealizar la función de probabilidad acumulada de la Distribución (Gumbel) de Valores Extremos:

# In[6]:


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


# Una vez que se hayan obtenido los mejores estimados para el parámetro de locación $(\lambda)$ y escala $(\delta)$, se puede estimar el tamaño máximo esperado de una característica microestructural, para una probabilidad dada, usando la ecuación:
# 
# $$x = \delta y + \lambda$$
# 
# o
# 
# $$x = \delta \left[-\ln\left(-\ln\left(P\right)\right)\right] + \lambda$$
# 
# Esto corresponde a la ecuación de una línea recta, en la que la pendiente está dada por el parámetro de escala $(\delta)$, y el intercepto está dado por el parámetro de locación $(\lambda)$.

# ### Construcción de un gráfico de Gumbel

# El desarrollo inicial del gráfico de Gumbel se llevó a cabo antes de la disponibilidad masiva de métodos computacionales. Es por eso, que este tipo de gráficos eran útiles para estimar mediante métodos gráficos los valores de los parámetros de la Distribución (Gumbel) de Valores Extremos. Al día de hoy, sirven como herramienta gráfica para verificar el buen ajuste de la distribución a los datos.
# 
# Los pasos para construír un gráfico de Gumbel son los siguientes:

# - Procesar la serie de datos utilizando la Distribución Empírica:
#     - Ordenar las mediciones de tamaño de característica microestructural, $x$, in orden ascendiente.
#     - Estimar los valores de la Distribución Empírica, $P$, para la serie de datos.
#     - Estimar el valor de la variable reducida que corresponde a la Distribución empírica ajustada previamente: $y=-\ln\left(-\ln\left(P\right)\right)$.

# - Utilizar un método para ajustar la Distribución (Gumbel) de Valores extremos a la serie de datos:
#     - Utilizar el método de momentos o el método de máxima verosimilitud para obtener estimados de los parámetros de la distribución, $\delta$ y $\lambda$.
#     - Obtener estimados de tamaños de la característica microestructural, $X_{ML}$, para las probabilidad empíricas estimadas para la serie de datos, $P$.

# - Graficar los datos:
#     - Graficar la distribución empírica linealizada contra las mediciones ordenadas, $-\ln\left(-\ln\left(P\right)\right)$ vs $x$.
#     - Graficar la distribución empírica linealizada contra las estimaciones dadas por la Distribución (Gumbel) de Valores Extremos, $-\ln\left(-\ln\left(P\right)\right)$ vs $x_{ML}$.
#     - Graficar los intervalos de confianza.

# La función `gumbelPlot()`, presenta una implementación del prodimiento para construír un gráfico de Gumbel.

# In[7]:


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


# # Ejemplo de aplicación de Análisis de Valores Extremos

# Los datos utilizados en este ejemplo se tomaron de la norma ASTM E-2283. Los datos están constituidos por 24 mediciones del tamaño más grande de inclusión observada en seis especímenes de acero (specimen: 1-6), en un procedimiento que se repite cuatro veces (run: A - D).
# 
# Debajo, los datos se ponen en un formato Pandas DataFrame, y se utiliza la función `gumbelPlot()` para ajustar la Distribución (Gumbel) de Valores Extremos a los datos, y trazar un gráfico de Gumbel para los mismos.

# In[8]:


# Datos de la norma ASTM-E2283
specimen = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
run = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D']
Y = [40.29, 37.24, 29.03, 52.46, 62.21, 33.98, 30.73, 37.43, 35.00, 44.82, 66.13, 48.55, 73.48, 44.79, 70.87, 59.83, 22.18, 64.32, 78.91, 46.53, 94.28, 49.15, 82.39, 37.43]


# In[9]:


# Construir el DataFrame
df = pd.DataFrame(list(zip(run, specimen, Y)),
                     columns=['run', 'specimen', 'Y'])


# In[10]:


# Mostrar el DataFrame
#df.sort_values(by='Y', ascending=True, inplace=True) # uncomment for sorting by size
display(df)


# In[11]:


# Construir el gráfico de Gumbel (con estimación de parámetros)
gumbelPlot(df)


# # Referencias

# - Norma ASTM: [ASTM E2283: Standard Practice for Extreme Value Analysis of Nonmetallic Inclusions in Steel and Other Microstructural Features](https://www.astm.org/e2283-08r19.html)
# 
# - Libro: [Y. Murakami, Metal Fatigue (2002)](https://www.sciencedirect.com/book/9780080440644/metal-fatigue)
# 
# - Página Wikipedia: [Gumbel Distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)
# 
# - Documentación Scipy: [scipy.stats.gumbel_r](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
