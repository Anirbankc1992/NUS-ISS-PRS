import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("LoanApplicant.csv")

# (1) Generate Summary Stats
sumry = np.round(data.describe().transpose(),decimals=2)

# (2) Histograms
data.hist(grid=True, figsize=(12,8), color='blue')
plt.tight_layout()
plt.show()

# (3) correlation matrices
corm = data.corr().values

# (4) Various Tests for PCA suitability
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value = calculate_bartlett_sphericity(data)

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data)


# (5) Tranform data
data_std = StandardScaler().fit_transform(data) 

# (6) Run the PCA Method
n_components = 8
pca = PCA(n_components).fit(data)

eigenvectors = np.round(pca.components_.transpose(),decimals=3)

eigenvalues = pca.explained_variance_

loadings= (np.sqrt(eigenvalues)*-1)*eigenvectors


