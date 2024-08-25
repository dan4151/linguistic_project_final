import numpy as np
from sklearn.linear_model import RidgeCV, MultiTaskLassoCV, MultiTaskElasticNetCV, LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from pygam import LinearGAM, s
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def learn_decoders(data, vectors, method):
    """
    Given data (a CxV matrix of V voxel activations per C concepts)
    and vectors (a CxD matrix of D semantic dimensions per C concepts),
    find a matrix M such that the dot product of M and a V-dimensional
    data vector gives a D-dimensional decoded semantic vector.

    Parameters:
    - data: CxV matrix of voxel activations
    - vectors: CxD matrix of semantic dimensions
    - method: string specifying the regression method to use (default is 'ridge')

    Returns:
    - M: Decoded semantic vector matrix (VxD)
    """

    method = method.lower()
    data = np.array(data)
    vectors = np.array(vectors)

    if method == 'ridge':
        model = RidgeCV(alphas=[1, 10, 0.1, 100, 0.01, 1000, 0.001, 10000, 0.0001, 100000, 0.00001, 1000000],
                        fit_intercept=False)
    elif method == 'svr':
        models = []
        for i in tqdm(range(vectors.shape[1]), desc="Training SVR models"):
            svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
            svr.fit(data, vectors[:, i])
            models.append(svr.coef_)
        return np.array(models).T.reshape(185866, vectors.shape[1])  # Transpose to match the expected shape
    elif method == 'pls':
        model = PLSRegression(n_components=10)
    elif method == 'pcr':
        pca = PCA(n_components=min(len(data), len(data[0])))
        data_reduced = pca.fit_transform(data)
        model = LinearRegression(fit_intercept=False)
        model.fit(data_reduced, vectors)
        return pca.inverse_transform(model.coef_).T


    model.fit(data, vectors)

    if hasattr(model, 'coef_'):
        return model.coef_.T
    elif hasattr(model, 'dual_coef_'):
        return model.dual_coef_.T
    else:
        raise ValueError("The chosen method does not have a coefficient attribute.")