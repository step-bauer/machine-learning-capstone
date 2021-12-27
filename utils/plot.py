import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Plot:
    
    @classmethod
    def print_null_dist (cls, df, filename=None, title=''):        
        fig, ax = plt.subplots(figsize=(8, 4))
        df_nulls = df.isnull().sum(axis=0) / df.shape[0]
        df_nulls['Dummy']=1
                
        ax.hist(df_nulls, bins =10, alpha=0.5)
        ax_bis = ax.twinx()
        
        ax.set_ylim(ymin=0,ymax=300)
        ax.set_xlim(xmin=-0.05,xmax=1.05)
        ax_bis.set_ylim(ymin=0,ymax=1.2)
        #ax_bis.set_xlim(xmin=0,xmax=1)
        

        
        ax_bis.hist(df_nulls, bins =50, cumulative=True, density=True, histtype='step', color='red', alpha=0.8, label='cum_line')
        
        plt.title(title)
        ax.set_xlabel('# % of missing values (NaN)')
        ax.set_ylabel('Columns');
        ax_bis.set_ylabel('cumulative');
        ax_bis.hlines(xmin=0, xmax=df_nulls.max(), y=0.9, linestyles='dashed', color='grey', label='0.9')
        ax_bis.legend(bbox_to_anchor=(1.07, 1.0), loc='upper left');

        if filename:
            plt.savefig(filename)
    
    @classmethod
    def plot_exp_var_ratio(cls, pca, width=10, height=7, vline_x=None, hline_y=None):
        """
        Description
        -----------
        
            Visualize the curves of the explained variance ratio for each component and the cumulative ratios

        Parameters
        ----------
            pca: sklearn.decomposition.pca.PCA
                PCA objct that is to be displayed
            
            width : float
                diagram width
                
            height : float
                diagram height

        Return
        ------
            None (visualizes a plot)

        """
        n_datapoints = len(pca.explained_variance_ratio_)            
        cumvals = np.cumsum(pca.explained_variance_ratio_)
        X = np.arange(0, n_datapoints)
        
        fig, ax = plt.subplots(1,1,figsize=(width, height))
        ax_bis  = ax.twinx()
        
        g1 = sns.lineplot(x= X, y=pca.explained_variance_ratio_, ax=ax, label='Variance')
        g2 = sns.lineplot(x=X, y=cumvals,ax=ax_bis, color='orange',label='Cumulative Variance', legend=False)
        
                
        if hline_y:
            ax_bis.hlines(y=hline_y,xmin=0, xmax=n_datapoints, linestyle='--', color='darkgrey')
        if vline_x:
            ax_bis.vlines(x=vline_x, ymin=0, ymax=1, linestyle='--', color='darkgrey')
        
        
        ax.set_xlabel("Components")
        ax.set_ylabel("Explained Variance Ratio %")
        ax.set_title("PCA Components Explained Variance Ratios")
        
        ax.set_yticks(np.arange(0, 0.21, step=0.01))
        ax.set_ylim(0,0.21)
        
        ax_bis.set_ylim(0,1.05)
        ax_bis.set_yticks(np.arange(0, 1.05, step=0.05))
        
        ax.set_xticks(np.arange(0, len(pca.explained_variance_ratio_)+2, step= (len(pca.explained_variance_ratio_) // width)))
        ax.grid(linewidth=1)
        
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_bis.get_legend_handles_labels()

        lines = lines_1 + lines_2
        labels = labels_1 + labels_2

        ax.legend(lines, labels, loc='upper left')
