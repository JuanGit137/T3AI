from logic import *
from dimreduct import reducesave_pca, reducesave_umap

for dim in dimensions:
    reducesave_pca(features, dim, get_pca_path(dim))
    reducesave_umap(features, dim, get_umap_path(dim))