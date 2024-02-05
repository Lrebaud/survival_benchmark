import numpy as np
import pandas as pd
# import swifter
# from swifter import set_defaults
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

# set_defaults(progress_bar=False)


class CustomOmicsImputer(BaseEstimator, TransformerMixin):

    def __init__(self, site_feature, min_frequency=0.1):
        self.site_feature = site_feature
        self.min_frequency = min_frequency
        self.imputer_ = None
        self.encoder_ = None
        self.len_encoding_ = None

    def fit(self, X, y=None):
        self.imputer_ = KNNImputer(n_neighbors=1)
        X = self.imputer_.fit_transform(X)
        self.encoder_ = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=self.min_frequency,
                                      sparse_output=False).fit(X[:, self.site_feature].reshape(-1, 1))
        return self

    def transform(self, X):
        X = self.imputer_.transform(X)
        b = self.encoder_.transform(X[:, self.site_feature].reshape(-1, 1))
        self.len_encoding_ = b.shape[1]
        a = np.delete(X, self.site_feature, 1)
        return np.hstack((a, b))


class RomaScoreReactome(BaseEstimator, TransformerMixin):

    def __init__(self, pathways, inter_threshold=0.9):
        self.pathways = pathways
        # self.gene_names = gene_names
        self.inter_threshold = inter_threshold
        self.gene_names = None
        self.pathways_filtered_ = None
        self.scaler_genes_ = None
        self.signatures_ = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "X should be a dataframe with gene names in columns"
        self.gene_names = X.columns
        self.scaler_genes_ = StandardScaler(with_std=False)
        data = pd.DataFrame(self.scaler_genes_.fit_transform(X.copy()).T, index=self.gene_names)

        filter_pathways = self.pathways.apply(
            lambda row: len(list(set(self.gene_names) & set(row))) / len(row) > self.inter_threshold)
        self.pathways_filtered_ = self.pathways[filter_pathways]
        # data = pd.DataFrame(self.scaler_genes_.fit_transform(X.copy()[:, :-1]).T, index = self.gene_names)

        def fun_pca(row):
            ind = list(set(self.gene_names) & set(row))
            # if len(ind)/len(row) >= self.threshold:
            pca = PCA(n_components=1, svd_solver="randomized")
            results = pd.Series(pca.fit_transform(data.loc[ind]).reshape(-1), index=ind)
            return results / pca.singular_values_[0]
            # return pd.Series(index = self.gene_names)

        # self.signatures_ = self.pathways_filtered_.swifter.apply(fun_pca).dropna(how='all', axis=0).reindex(
        #     columns=self.gene_names).fillna(value=0).T
        self.signatures_ = self.pathways_filtered_.apply(fun_pca).dropna(how='all', axis=0).reindex(
            columns=self.gene_names).fillna(value=0).T
        return self

    def transform(self, X):
        return np.dot(self.scaler_genes_.transform(X.copy()), self.signatures_.values)


class OmicsPreprocessor(TransformerMixin):

    def __init__(self, counts_threshold, gene_names):
        self.counts_threshold = counts_threshold
        self.gene_names = gene_names
        self.tuple_genes = pd.MultiIndex.from_frame(self.gene_names)
        self.selected_genes_ = None

    def fit(self, X, y=None):
        # 1. Transform data into pandas dataframe for preprocessing operations
        temp = pd.DataFrame(X.copy().T, index=self.tuple_genes)

        # 2. Filter low-expressed genes
        temp_filtered = temp.loc[(temp > 0).sum(axis=1) > self.counts_threshold*temp.shape[1]]

        # 3. Drop duplicates based on lowest standard deviation
        def aggregate_dup(data):
            data['std'] = np.log(data.copy().drop(columns=["gene_name", "gene_id"]) + 1).std(axis=1)
            data = (data.sort_values(by='std', ascending=False)
                    .groupby(axis=0, by="gene_name")
                    .first()
                    .drop(columns='std')
                    )
            return data

        temp_filtered = aggregate_dup(temp_filtered.reset_index(level=["gene_id", "gene_name"]))
        self.selected_genes_ = temp_filtered.set_index("gene_id").index.values
        return self

    def transform(self, X):
        # 1. Transform data into pandas dataframe
        temp = (pd.DataFrame(X.copy().T, index=self.tuple_genes).reset_index(level="gene_name", drop=False)
                .loc[self.selected_genes_])
        return np.log(temp.set_index("gene_name", drop=True) + 1).T


def load_RNAseq(data_path, table_annot_path, table_corr_path):
    table_annot = pd.read_csv(table_annot_path, index_col=0).set_index('gene_id')
    table_corr = pd.read_csv(table_corr_path, index_col=0, sep=';').iloc[:, :2]

    list_df = []
    for path in data_path:
        list_df.append(pd.read_csv(path, index_col=0, sep=";").rename(columns=table_corr["Patient Name"]))

    df_RNA = pd.concat(list_df, axis=1, join='inner')
    gene_names = table_annot.loc[df_RNA.index].reset_index().rename(columns={"index": "gene_id"})

    return df_RNA.T, gene_names


def load_filter_pathways(path, co_table_path, min_pathway=10, max_pathway=500):
    co_table = pd.read_csv(co_table_path, index_col=0)
    pathways = pd.read_csv(path,
                           sep="\t",
                           header=0,
                           names=["EnsemblID", "Pathway", "URL", "Description", "Index", "Species"])
    pathways_filtered = pathways[(pathways['Species'] == "Homo sapiens") & (pathways['Index'] == "TAS")]

    def f(df):
        return co_table.loc[[ens.split('.')[0] for ens in df['EnsemblID'].values]]["0"].dropna().unique()

    # filter pathway size
    reactome = pathways_filtered.groupby("Pathway").apply(f)
    temp = reactome.apply(len)

    return reactome[(temp >= min_pathway) & (temp <= max_pathway)]
