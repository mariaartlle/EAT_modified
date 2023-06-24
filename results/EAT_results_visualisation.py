import pickle, h5py, json, statistics
from urllib.request import urlopen
from pyexpat import features
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as matplt
import matplotlib.cm as cm

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
# clustering methods imports 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# visualization imports
# from umap import UMAP
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from pacmap import PaCMAP

def extract_tsv_from_fasta(fasta):
    df = pd.DataFrame(columns=['UniprotID', 'Origin_database', 'Family', 'Description', 'Sequence'])
    i = 1
    with open(fasta, 'r') as file:
        for line in file:
            if line.startswith('>'):
                identifier = line.strip('>\n').split('|')
                df.at[i, 'UniprotID'] = identifier[0]
                df.at[i, 'Origin_database'] = identifier[2]
                df.at[i, 'Family'] = identifier[3]
                df.at[i, 'Description'] = identifier[4]
            else:
                sequence = line.strip()
                df.at[i, 'Sequence'] = sequence
                i += 1

    df.to_csv('BacDBTF_full.tsv', sep='\t', index=False)

def generate_fastas(tsv):
    df = pd.read_csv(tsv, sep='\t')
    famdict = {}
    for family in set(df['Family'].to_list()):
        proteins = list(df.loc[df['Family'] == family, 'UniprotID'])
        famdict[family] = proteins

    n_protxfam = {i:len(j) for i, j in famdict.items()}

    for n in [5, 50]:
        fam = [k for k, v in n_protxfam.items() if v > n]
        df1 = pd.DataFrame()
        for family in fam:
            subset = df.loc[df['Family'] == family]
            df1 = pd.concat([df1, subset])

        df1.reset_index(inplace=True)
        df1.to_csv('BacDBTF_morethan{}.tsv'.format(n), sep='\t', index=False)
        with open('BacDBTF_morethan{}.fasta'.format(n), 'w') as fasta:
            for i in range(len(df1.axes[0])):
                fasta.write('>{}_{}\n{}\n'.format(df1.at[i, 'UniprotID'], df1.at[i, 'Family'], df1.at[i, 'Sequence']))

def agglomerative_clustering(df, embeddings, labels_true, n_clusters, method='tSNE'):
    embeddings = StandardScaler().fit_transform(embeddings)
    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    labels = ag.labels_
    df['cluster_labels'] = labels
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))

    if method == 'PaCMAP':
        reduced = PaCMAP(n_components=2, random_state=123, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['PaCMAP 1', 'PaCMAP 2'])
    elif method == 'UMAP':
        reduced = UMAP(random_state=123).fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['UMAP 1', 'UMAP 2'])
    elif method == 'PCA': 
        reducer = PCA(n_components=2, random_state=123)
        reduced = reducer.fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['PCA 1', 'PCA 2'])
    elif method == 'tSNE':
        if len(embeddings) <= 30:
            reduced = TSNE(n_jobs=-1, random_state=123, perplexity=(len(embeddings)-1)).fit_transform(embeddings)
        else:
            reduced = TSNE(n_jobs=-1, random_state=123, perplexity=40).fit_transform(np.array(embeddings))
        reduced_df = pd.DataFrame(reduced, columns = ['tSNE 1', 'tSNE 2'])
    else:
        raise Exception(f'Dimensionality reduction method {method} needs to be either PCA, tSNE, UMAP or PaCMAP')
    
    df = pd.concat([df, reduced_df], axis='columns')    
    return df, metrics.v_measure_score(labels_true, labels)

def kmeans_clustering(df, embeddings, labels_true, n_clusters, method='PaCMAP'):
    embeddings = StandardScaler().fit_transform(embeddings)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=4).fit(embeddings)
    labels = kmeans.labels_
    df['cluster_labels'] = labels
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    
    if method == 'PaCMAP':
        reduced = PaCMAP(n_components=2, random_state=123, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['PaCMAP 1', 'PaCMAP 2'])
    elif method == 'UMAP':
        reduced = UMAP(random_state=123).fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['UMAP 1', 'UMAP 2'])
    elif method == 'PCA': 
        reducer = PCA(n_components=2, random_state=123)
        reduced = reducer.fit_transform(embeddings)
        reduced_df = pd.DataFrame(reduced, columns = ['PCA 1', 'PCA 2'])
    elif method == 'tSNE':
        if len(embeddings) <= 30:
            reduced = TSNE(n_jobs=-1, random_state=123, perplexity=(len(embeddings)-1)).fit_transform(embeddings)
        else:
            reduced = TSNE(n_jobs=-1, random_state=123, perplexity=40).fit_transform(np.array(embeddings))
        reduced_df = pd.DataFrame(reduced, columns = ['tSNE 1', 'tSNE 2'])
    else:
        raise Exception(f'Dimensionality reduction method {method} needs to be either PCA, tSNE, UMAP or PaCMAP')
    
    df = pd.concat([df, reduced_df], axis='columns')  
    return df, metrics.v_measure_score(labels_true, labels)

def run_benchmark(data, labels, savingpath='.', name='results'):
    def bench_k_means(kmeans, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        t0 = time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        fit_time = time() - t0
        # results = [name, fit_time, estimator[-1].inertia_]
        results = [fit_time, estimator[-1].inertia_]

        # Define the metrics which require only the true labels and estimator
        # labels
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # The silhouette score requires the full dataset
        results += [
            metrics.silhouette_score(
                data,
                estimator[-1].labels_,
                metric="euclidean",
                sample_size=300,
            )
        ]

        # Show the results
        formatter_result = (
            "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
        )
        formatted_result = ['{:.3f}'.format(x) for x in results]
        # print(formatter_result.format(*results))
        return pd.Series(formatted_result, index=['time', 'inertia', 'homo', 'compl', 'v-means', 'ARI', 'AMI', 'silhouette'])

    def bench_agglomerative_clustering(ag, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        ag : AgglomerativeClustering instance
            A :class:`~sklearn.cluster.AgglomerativeClustering` instance with the initialization
            already set.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        t0 = time()
        estimator = make_pipeline(StandardScaler(), ag).fit(data)
        fit_time = time() - t0
        # results = [name, fit_time, estimator[-1].inertia_]
        results = [fit_time, 0]

        # Define the metrics which require only the true labels and estimator
        # labels
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # The silhouette score requires the full dataset
        results += [
            metrics.silhouette_score(
                data,
                estimator[-1].labels_,
                metric="euclidean",
                sample_size=300,
            )
        ]
        # Show the results
        formatted_result = ['{:.3f}'.format(x) for x in results]
        # print(formatter_result.format(*results))
        return pd.Series(formatted_result, index=['time', 'inertia', 'homo', 'compl', 'v-means', 'ARI', 'AMI', 'silhouette'])

    df = pd.DataFrame()
    for n in range(25, 46):
        series_dict = {}
        print('n_clusters: {}'.format(n))
        # print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

        # kmeans = KMeans(n_clusters=n, n_init='auto', random_state=10)
        # series_dict["k-means_auto"] = bench_k_means(kmeans=kmeans, name="k-means_auto", data=data, labels=labels)

        kmeans = KMeans(init="k-means++", n_clusters=n, n_init=4, random_state=10)
        series_dict["k-means++"] = bench_k_means(kmeans=kmeans, data=data, labels=labels)

        ag = AgglomerativeClustering(n_clusters=n)
        series_dict['aggl_clust'] = bench_agglomerative_clustering(ag, data=data, labels=labels)

        # kmeans = KMeans(init="random", n_clusters=n, n_init=4, random_state=10)
        # bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)
        # series_dict["random"] = bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

        # pca = PCA(n_components=n).fit(data)
        # kmeans = KMeans(init=pca.components_, n_clusters=n, n_init=1)
        # bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)
        # series_dict["PCA-based"] = bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

        multiindex = pd.MultiIndex.from_tuples([(f'{n}', 'k-means++'), (f'{n}', 'aggl_clust')], names=['n_clusters', 'algorithm'])
        dictdf = pd.DataFrame(series_dict)
        dictdf.columns = multiindex
        df = pd.concat([df, dictdf], axis=1)
        del dictdf
    # df.to_csv('{}/{}_benchmarking_clustering.tsv'.format(savingpath, name), sep='\t')
    df.to_csv('{}_benchmarking_clustering.tsv'.format(name), sep='\t')

def visualize_embeddings(features_df, path4figures= '.', model_used='35M', clustering_method='kmeans', name='Col', vmeasure=None):

    col1 = '{} 1'.format(name)
    col2 = '{} 2'.format(name)

    # fig1, ax1 = plt.subplots(1, figsize = (10,10))
    fig2, ax2 = plt.subplots(1, figsize = (10,10))
    fig3, ax3 = plt.subplots(1, figsize = (10,10))

    x = features_df[col1].values
    y = features_df[col2].values

 
    color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    # cmap = cm.get_cmap('Set1', 38)
    # color_list = [matplt.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    
    clustercolors = {} 
    i = 0
    for cluster in set(features_df['cluster_labels'].to_list()):
        if str(cluster) != 'nan':
            clustercolors[int(cluster)] = color_list[i]
            i += 1

    familycolors = {}
    i = 0
    for family in set(features_df['Family'].to_list()):
        if str(family) != 'nan':
            familycolors[str(family)] = color_list[i]
            i += 1

    def familycolormask(mylist, familycolors):
        mylist = [familycolors[str(x)] for x in mylist if x in familycolors.keys()]
        return mylist

    def clustercolormask(mylist, familycolors):
        mylist = [familycolors[x] for x in mylist if x in familycolors.keys()]
        return mylist


    for family in set(features_df['Family'].values):
        subset_ref_family = features_df.loc[features_df['Family'] == family]
        xref_family = subset_ref_family[col1].values
        yref_family = subset_ref_family[col2].values
        ax2.scatter(xref_family, yref_family, c = familycolormask(subset_ref_family['Family'].to_list(), familycolors), alpha = 0.5, label = '{}'.format(family))

    for cluster in set(features_df['cluster_labels'].values):
        subset_ref_cluster = features_df.loc[features_df['cluster_labels'] == cluster]
        xref_family = subset_ref_cluster[col1].values
        yref_family = subset_ref_cluster[col2].values
        ax3.scatter(xref_family, yref_family, c = clustercolormask(subset_ref_cluster['cluster_labels'].to_list(), clustercolors), alpha = 0.5, label = '{}'.format(cluster))
    # ax3.scatter(features_df[col1].values, features_df[col2].values, c = 'blue', alpha = 0.5, label = 'KPA171202')

    # for i, txt in enumerate(features_df['UniprotID'].to_list()):
    #     if txt in ['Q6AAR7', 'Q6A6D9', 'Q6A638']:
    #         ax3.annotate(txt, (x[i], y[i]), fontweight='bold', arrowprops=dict(arrowstyle='->'))
    #     else:
    #         ax3.annotate(txt, (x[i], y[i]))

    # ax3.scatter(x, y, c=clustercolormask(features_df['cluster_labels'].to_list()), alpha=0.5, label = 'reference')


    ax2.grid(True)
    ax3.grid(True)

    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Families")
    # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Clusters')
    ax2.set_title('V-measure: {}'.format(vmeasure))
    ax3.set_title('V-measure: {}'.format(vmeasure))

    for a in [ax2, ax3]:
        a.set_xlabel(col1)
        a.set_ylabel(col2)

    fig2.savefig('ESM2_{}_{}_{}_family.png'.format(model_used, clustering_method, name) , format = 'png', dpi = 300)
    fig3.savefig('ESM2_{}_{}_{}_clusters.png'.format(model_used, clustering_method, name) , format = 'png', dpi = 300)

    return features_df.drop_duplicates()

def visualize_embeddings_w_acnes_highlighted(features_df, acnes_uniprots, path4figures= '.', model_used='35M', clustering_method='kmeans', name='Col', vmeasure=None):

    col1 = '{} 1'.format(name)
    col2 = '{} 2'.format(name)

    # fig1, ax1 = plt.subplots(1, figsize = (10,10))
    # fig2, ax2 = plt.subplots(1, figsize = (10,10))
    fig3, ax3 = plt.subplots(1, figsize = (10,10))

    x = features_df[col1].values
    y = features_df[col2].values

 
    color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']
    # cmap = cm.get_cmap('Set1', 38)
    # color_list = [matplt.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    
    clustercolors = {} 
    i = 0
    for cluster in set(features_df['cluster_labels'].to_list()):
        if str(cluster) != 'nan':
            clustercolors[int(cluster)] = color_list[i]
            i += 1

    # familycolors = {}
    # i = 0
    # for family in set(features_df['Family'].to_list()):
    #     if str(family) != 'nan':
    #         familycolors[str(family)] = color_list[i]
    #         i += 1

    def familycolormask(mylist, familycolors):
        mylist = [familycolors[str(x)] for x in mylist if x in familycolors.keys()]
        return mylist

    def clustercolormask(mylist, familycolors):
        mylist = [familycolors[x] for x in mylist if x in familycolors.keys()]
        return mylist


    # for family in set(features_df['Family'].values):
    #     subset_ref_family = features_df.loc[features_df['Family'] == family]
    #     xref_family = subset_ref_family[col1].values
    #     yref_family = subset_ref_family[col2].values
    #     ax2.scatter(xref_family, yref_family, c = familycolormask(subset_ref_family['Family'].to_list(), familycolors), alpha = 0.5, label = '{}'.format(family))
    
    # get the cluster most populated by acnes to remove annotations
    subset_acnes = features_df.loc[features_df['Family'] == 'acnes']
    acnes_clusters = {i:subset_acnes['cluster_labels'].to_list().count(i) for i in set(subset_acnes['cluster_labels'].to_list())}
    cluster_acnes, _ = list(sorted(acnes_clusters.items(), key=lambda x: x[1]))[-1]


    for cluster in set(features_df['cluster_labels'].values):
        subset_ref_cluster = features_df.loc[features_df['cluster_labels'] == cluster]
        xref_family = subset_ref_cluster[col1].values
        yref_family = subset_ref_cluster[col2].values
        ax3.scatter(xref_family, yref_family, c = clustercolormask(subset_ref_cluster['cluster_labels'].to_list(), clustercolors), alpha = 0.5, label = '{}'.format(cluster))
    # ax3.scatter(features_df[col1].values, features_df[col2].values, c = 'blue', alpha = 0.5, label = 'KPA171202')

    for i, txt in enumerate(features_df['UniprotID'].to_list()):
        if txt in acnes_uniprots:
            # ax3.annotate(txt, (x[i], y[i]), fontweight='bold', arrowprops=dict(arrowstyle='->'))
            cluster_label = list(features_df.loc[features_df['UniprotID'] == txt, 'cluster_labels'])[0]
            ax3.scatter(x[i], y[i], c = clustercolormask([cluster_label], clustercolors), edgecolors='black')
            if str(cluster_label) != str(cluster_acnes):
                ax3.annotate(txt, (x[i], y[i]), color='black')
                

    # ax3.scatter(x, y, c=clustercolormask(features_df['cluster_labels'].to_list()), alpha=0.5, label = 'reference')


    # ax2.grid(True)
    ax3.grid(True)

    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Families")
    # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Clusters')
    # ax2.set_title('V-measure: {}'.format(vmeasure))
    # ax3.set_title('V-measure: {}'.format(vmeasure))

    # for a in [ax2, ax3]:
    for a in [ax3]:
        a.set_xlabel(col1)
        a.set_ylabel(col2)

    # fig2.savefig('ESM2_{}_{}_{}_family.png'.format(model_used, clustering_method, name) , format = 'png', dpi = 300)
    fig3.savefig('ESM2_BacDBTF_acnes_{}_{}_{}_clusters.png'.format(model_used, clustering_method, name) , format = 'png', dpi = 300)

    return features_df.drop_duplicates()

def BacDBTF_embeddings(embeddings_path, acnes_embeddings_dict='', clustering='kmeans'):
    # model = embeddings_path.split('/')[-2]
    model = '650M'
    path = '/'.join(embeddings_path.split('/')[:-1])

    if str(embeddings_path)[-3:] == 'csv':
        method = 'tSNE'
        # for method in ['PCA', 'tSNE', 'PaCMAP', 'UMAP']:
        data = pd.read_csv(embeddings_path)
        df = pd.DataFrame(data)
        
        df['UniprotID'] = df['Label'].apply(lambda x: x.split('_')[0])
        df['Family'] = df['Label'].apply(lambda x: x.split('_')[1])
        if acnes_embeddings_dict != '':
            embeddings = list(df.iloc[:,:-3].values) + list(acnes_embeddings_dict.values())
        else:
            embeddings = df.iloc[:,:-3].values

        families = df.iloc[:,-1].values

        # df1 = pd.read_csv('/home/maria/acnes_sensors/AI/ESM2/acnes_embeddoma/acnes_annotations.tsv', sep='\t', index_col=False)
        # df1 = df1[['old_locus_tag', 'protein_description']]

        # df = pd.merge(df1, df, on='old_locus_tag')
        # print(df)
        # Cluster the embeddings 
        if clustering == 'kmeans':
            df, vmeasure = kmeans_clustering(df, embeddings, families, 38, method=method)
        elif clustering == 'agglomerative':
            df, vmeasure = agglomerative_clustering(df, embeddings, families, 38, method=method)

        df = visualize_embeddings(df, path4figures=path, model_used=model, clustering_method=clustering, name=method, vmeasure=vmeasure)
        col1 = method+' 1'
        col2 = method+' 2'
        df = df[['UniprotID', 'Family', 'cluster_labels', col1, col2]]
        df.to_csv('{}/BacDBTF_{}_{}_merged.tsv'.format(path, clustering, model), sep='\t', index=False)

    elif str(embeddings_path)[-3:] == 'pkl':
        # set the acnes proteins as their one family 
        method = 'tSNE'
        if acnes_embeddings_dict != '':
            families = ['acnes' for i in range(len(acnes_embeddings_dict.keys()))]
            acnes_uniprots = list(acnes_embeddings_dict.keys())

            df = pd.DataFrame()
            with open(embeddings_path, "rb") as input_file:
                embeddings = pickle.load(input_file)['protein_embs']
                for uniprot, embed in embeddings.items():
                        # there is some redundancy, so only add family if the uniprot is not in the dict already (if not the length of arrays is not the same)
                        if uniprot.split('_')[0].strip('>') not in acnes_embeddings_dict.keys():
                            families.append(uniprot.split('_')[1])
                        acnes_embeddings_dict[uniprot.split('_')[0].strip('>')] = embed
            
            embeddings = list(acnes_embeddings_dict.values())
            df['UniprotID'] = acnes_embeddings_dict.keys()
            df['Family'] = families
            if clustering == 'kmeans':
                df, vmeasure = kmeans_clustering(df, embeddings, families, 44, method=method)
            elif clustering == 'agglomerative':
                df, vmeasure = agglomerative_clustering(df, embeddings, families, 44, method=method)

            df = visualize_embeddings_w_acnes_highlighted(df, acnes_uniprots, path4figures=path, model_used=model, clustering_method=clustering, name=method, vmeasure=vmeasure)
            col1 = method+' 1'
            col2 = method+' 2'
            df = df[['UniprotID', 'Family', 'cluster_labels', col1, col2]]
            # df.to_csv('{}/BacDBTF_acnes_{}_{}_merged.tsv'.format(path, clustering, model), sep='\t', index=False)
            df.to_csv('BacDBTF_acnes_{}_{}_merged.tsv'.format(clustering, model), sep='\t', index=False)
        else:
            families = []
            embeddings_dict = dict()
            df = pd.DataFrame()
            with open(embeddings_path, "rb") as input_file:
                embeddings = pickle.load(input_file)['protein_embs']
                for uniprot, embed in embeddings.items():
                        # there is some redundancy, so only add family if the uniprot is not in the dict already (if not the length of arrays is not the same)
                        if uniprot.split('_')[0].strip('>') not in embeddings_dict.keys():
                            families.append(uniprot.split('_')[1])
                        embeddings_dict[uniprot.split('_')[0].strip('>')] = embed
            
            embeddings = list(embeddings_dict.values())
            df['UniprotID'] = embeddings_dict.keys()
            df['Family'] = families
            if clustering == 'kmeans':
                df, vmeasure = kmeans_clustering(df, embeddings, families, 44, method=method)
            elif clustering == 'agglomerative':
                df, vmeasure = agglomerative_clustering(df, embeddings, families, 44, method=method)

            df = visualize_embeddings(df, path4figures=path, model_used=model, clustering_method=clustering, name=method, vmeasure=vmeasure)
            col1 = method+' 1'
            col2 = method+' 2'
            df = df[['UniprotID', 'Family', 'cluster_labels', col1, col2]]
            df.to_csv('BacDBTF_{}_{}_merged.tsv'.format(clustering, model), sep='\t', index=False)


    # run_benchmark(embeddings, families, savingpath=path, name='{}'.format(model))

def h5_to_dict(h5f):
    h5_f = h5py.File(h5f, 'r')

    dataset = {}
    for key in h5_f.keys():
        uniprot = key.split('_') [0]
        dataset[uniprot] = h5_f[key][()]
    return dataset

def prottucker_embedding_space_visualisation_by_query(query_h5, lookup_h5, validation=False): 
    '''
    Generate a visualisation plot colored by the type of dataset query/validation vs. BacDBTF/training data
    '''
    if validation == True: 
        query_type = 'Validation data'
        lookup_type = 'Training data'
    else:
        query_type = 'C. acnes'
        lookup_type = 'BacDBTF'

    query_emb = h5_to_dict(query_h5)
    lookup_emb = h5_to_dict(lookup_h5)

    full_emb = {**query_emb,**lookup_emb}

    # Dimensionality reduction
    reduced = TSNE(n_jobs=-1, random_state=123, perplexity=40).fit_transform(np.array(list(full_emb.values())))
    df = pd.DataFrame(reduced, columns = ['tSNE 1', 'tSNE 2'])
    df['UniprotID'] = full_emb.keys()
    df['Type'] = ''

    for i in range(len(df.axes[0])): 
        if df.at[i, 'UniprotID'] in query_emb.keys(): 
            df.at[i, 'Type'] = query_type
        else: 
            df.at[i, 'Type'] = lookup_type

    fig, ax = plt.subplots(1, figsize = (10,10))

    x = df['tSNE 1'].values
    y = df['tSNE 2'].values

    color_dict = {
        lookup_type: 'cyan',
        query_type : 'magenta'
    }

    for c in color_dict.keys():
        subset = df.loc[df['Type'] == c]
        subx = subset['tSNE 1'].values
        suby = subset['tSNE 2'].values
        ax.scatter(subx, suby, c=color_dict[c], alpha = 0.5, label='{}'.format(c))
    
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Type of protein')
    ax.set_title('Contrastive learning implementation')
    if validation == True:
        fig.savefig('EAT_validation_training_space.png', format='png', dpi=300)
    else:
        fig.savefig('EAT_acnes_inference_training.png', format='png', dpi=300)

def prottucker_embeddings_byfamily(query_h5, lookup_h5): 
    '''
    Generate a visualisation plot colored by the family label
    '''
    query_emb = h5_to_dict(query_h5)
    lookup_emb = h5_to_dict(lookup_h5)


    full_emb = {**query_emb,**lookup_emb}

    # Dimensionality reduction
    reduced = TSNE(n_jobs=-1, random_state=123, perplexity=40).fit_transform(np.array(list(full_emb.values())))
    df = pd.DataFrame(reduced, columns = ['tSNE 1', 'tSNE 2'])
    df['UniprotID'] = full_emb.keys()
    df['Family'] = ''

    x = df['tSNE 1'].values
    y = df['tSNE 2'].values

    annotated_proteins = ['Q6A6R1', 'Q6A6K7', 'Q6A8H5', 'Q6A8Y2', 'Q6A8A1', 'Q6A998', 'Q6A7V9', 'Q6AA75']

    # merge the embeddings with the family labels by UniprotID
    labelsdf = pd.read_csv('/home/maria/EAT_modified/data/BacDBTF/lookup_queries.csv', header=None)
    labels_dict = {i:j for i, j in zip(labelsdf[0], labelsdf[1])}

    for i in range(len(df.axes[0])): 
        if df.at[i, 'UniprotID'] in query_emb.keys(): 
            df.at[i, 'Family'] = 'C. acnes'
        else: 
            df.at[i, 'Family'] = labels_dict[df.at[i, 'UniprotID']]

    fig, ax = plt.subplots(1, figsize = (10,10))

    color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']

    familycolors = {}
    i = 0
    for family in set(df['Family'].to_list()):
        if str(family) == 'C. acnes': 
            familycolors[str(family)] = '#000000'
        elif str(family) != 'nan':
            familycolors[str(family)] = color_list[i]
        i += 1

    def familycolormask(mylist, familycolors):
        mylist = [familycolors[str(x)] for x in mylist if x in familycolors.keys()]
        return mylist

    for family in set(df['Family'].values):
        subset_ref_family = df.loc[df['Family'] == family]
        xref_family = subset_ref_family['tSNE 1'].values
        yref_family = subset_ref_family['tSNE 2'].values
        ax.scatter(xref_family, yref_family, c = familycolormask(subset_ref_family['Family'].to_list(), familycolors), alpha = 0.5, label = '{}'.format(family))
    
    for i, txt in enumerate(df['UniprotID'].to_list()):
        family_label = list(df.loc[df['UniprotID'] == txt, 'Family'])[0]
        if txt in annotated_proteins: 
            ax.scatter(x[i], y[i], c = )



    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Families')

    fig.savefig('EAT_acnes_inference_by_family.png', format='png', dpi=300)

def retrieve_uniprot_annotation(eat_results_tsv): 

    df = pd.read_csv(eat_results_tsv, sep='\t')
    df['Lookup-uniprot-annotation'] = ''

    for i in range(len(df.axes[0])):
        uniprot = df.at[i, 'Query-ID']
        url = 'https://rest.uniprot.org/uniprotkb/{}.json'.format(uniprot)
        response = urlopen(url)
        data_json = json.loads(response.read())
        try:
            df.at[i, 'Query-Label'] = data_json['proteinDescription']['recommendedName']['fullName']['value']
        except: 
            df.at[i, 'Query-Label'] = 'No annotation'

    for i in range(len(df.axes[0])):
        uniprot = df.at[i, 'Lookup-ID']
        url = 'https://rest.uniprot.org/uniprotkb/{}.json'.format(uniprot)
        response = urlopen(url)
        data_json = json.loads(response.read())
        try:
            df.at[i, 'Lookup-uniprot-annotation'] = data_json['proteinDescription']['recommendedName']['fullName']['value']
        except: 
            df.at[i, 'Lookup-uniprot-annotation'] = 'No annotation'

    df.to_csv('eat_results_mapped.tsv', sep='\t', index=False)

def calculate_mean_distances(eat_results_tsv):
    df = pd.read_csv(eat_results_tsv, sep='\t', index_col=False)
    df['Dataset'] = ''
    annotated_proteins = ['Q6A6R1', 'Q6A6K7', 'Q6A8H5', 'Q6A8Y2', 'Q6A8A1', 'Q6A998', 'Q6A7V9', 'Q6AA75']
    df['Annotation'] = df['Query-ID'].apply(lambda x: 'Agreement with Uniprot' if x in annotated_proteins else 'Not annotated')
    embedding_distances = df['Embedding distance'].to_list()
    mean = sum(embedding_distances)/ len(embedding_distances)

    print('The mean distance of all proteins and their embeddings is: {}'.format(mean))
    print('The median of all distances: {}'.format(statistics.median(embedding_distances)))
    print('The standard deviation of the mean of all the distances: {}'.format(statistics.pstdev(embedding_distances)))

    annotated_distances = []
    for i in range(len(df.axes[0])):
        uniprot = df.at[i, 'Query-ID']
        if uniprot in annotated_proteins: 
            annotated_distances.append(float(df.at[i, 'Embedding distance']))

    mean_annotated = sum(annotated_distances) / len(annotated_distances)

    print('The mean distance of annotated proteins and their embeddings is: {}'.format(mean_annotated))
    print('The median of annotated distances: {}'.format(statistics.median(annotated_distances)))
    print('The standard deviation of the mean of annotated distances: {}'.format(statistics.pstdev(annotated_distances)))

    # violin/scatter plot that shows the distribution of embedding distances in the dataset with the annotated proteins highlighted and annptated in the graphs

    fig, ax = plt.subplots(1, figsize = (10,10))
      
    ax = sns.violinplot(data=df, x="Dataset", y="Embedding distance", inner=None, linewidth=0, scale="count", color='#9BDEAC')
    ax = sns.swarmplot(data=df, x="Dataset", y="Embedding distance", hue='Annotation', palette={'Agreement with Uniprot': '#255957', 'Not annotated': '#C33149'})
    
    # for i, txt in enumerate(df['Query-ID'].to_list()):
    #     if txt in annotated_proteins:
    #         # ax.scatter(0, embedding_distances[i], color='#437C90', alpha=0.5)
    #         ax.annotate(txt, (0,embedding_distances[i]))

    # ax.set_yticks(range()))
    ax.set_yticks(np.arange(0, 0.8, 0.05))
    for violin, alpha in zip(ax.collections[::2], [0.2]):
        violin.set_alpha(alpha)

    fig.savefig('violin_scatter_CL_distances.png', format='png', dpi=300)

if __name__ == '__main__':

    # validation_h5 = '/home/maria/EAT_modified/results/validation_prottucker_embeddings.h5'
    training_h5 = '/home/maria/EAT_modified/results/BacDBTF50_ESM2_wo_acnes/training_prottucker_embeddings.h5'
    query_h5 = '/home/maria/EAT_modified/results/BacDBTF50_ESM2_wo_acnes/query_prottucker_embeddings.h5'
    # lookup_h5 = '/home/maria/EAT_modified/results/lookup_prottucker_embeddings.h5'
    # eat_results_tsv = '/home/maria/EAT_modified/results/BacDBTF50_ESM2_wo_acnes/eat_results_mapped.tsv'


    # prottucker_embedding_space_visualisation_by_query(query_h5, training_h5, validation=False)
    prottucker_embeddings_byfamily(query_h5, training_h5)

    # retrieve_uniprot_annotation(eat_results_tsv)
    # calculate_mean_distances(eat_results_tsv)