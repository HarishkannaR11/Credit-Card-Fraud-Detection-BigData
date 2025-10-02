import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_features(X):
    logging.info("Scaling features")
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_pca(X, n_components=2):
    logging.info(f"Applying PCA with {n_components} components")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)
