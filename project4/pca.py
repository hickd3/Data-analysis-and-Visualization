'''pca_cov.py
Performs principal component analysis using the covariance matrix of the dataset
Dean Hickman
CS 251 / 252: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_transformations import normalize, center


class PCA:
    '''Perform and store principal component analysis results

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.eig`
    The numpy functions that you have been using so far are fine to use.
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_mins: ndarray. shape=(num_selected_vars,)
        #   Mins of each orignal data variable
        self.orig_mins = None

        # orig_maxs: ndarray. shape=(num_selected_vars,)
        #   Maxs of each orignal data variable
        self.orig_maxs = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        centered_data = center(data)
        cov_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0]-1)
        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        total_variance = np.sum(e_vals)
        prop_var = e_vals / total_variance
        return prop_var.tolist()

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = np.cumsum(prop_var)
        return cum_var.tolist()

    def fit(self, vars, normalize_dataset=False):
        '''Fits PCA to the data variables `vars` by computing the full set of PCs. The goal is to compute 
        - eigenvectors and eigenvalues
        - proportion variance accounted for by each PC.
        - cumulative variance accounted for by first k PCs.
        
        Does NOT actually transform data by PCA.

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        HINT:
        - It may be easier to convert to numpy ndarray format once selecting the appropriate data variables.
        - Before normalizing (if normalize_dataset is true), create instance variables containing information that would
        be needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        - Remember, this method does NOT actually transform the dataset by PCA.
        '''
        self.vars = vars
        self.A = self.data[vars].to_numpy()

        self.orig_mins = self.A.min(axis=0)
        self.orig_maxs = self.A.max(axis=0)
        print("mins before normalization:", self.orig_mins)
        print("maxs before normalization:", self.orig_maxs)

        if normalize_dataset:
            self.A = (self.A - self.orig_mins) / (self.orig_maxs - self.orig_mins)
            print("expected min:", self.A.min(axis=0))
            print("expected max:", self.A.max(axis=0))

        print("data min/max", self.A.min(), "/", self.A.max())
        
        cov_matrix = self.covariance_matrix(self.A)
        self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)
        index_sorted = np.argsort(self.e_vals)[::-1]
        self.e_vals = self.e_vals[index_sorted]
        self.e_vecs = self.e_vecs[:, index_sorted]
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if self.cum_var is None:
            raise ValueError('Cant plot cumulative variance. Compute the PCA first.')

        if num_pcs_to_keep == None:
            PCs = np.arange(len(self.cum_var)+1)
            proportions = np.zeros(len(self.cum_var)+1)
            proportions[1:] = self.cum_var
            plt.plot(PCs, proportions, marker = '.', c = 'k', markersize = 15)
        else:
            PCs = np.arange(num_pcs_to_keep+1)
            proportions = np.zeros(num_pcs_to_keep+1)
            proportions[1:] = self.cum_var[:num_pcs_to_keep]
            plt.plot(PCs, proportions, marker = '.', c = 'k', markersize = 15)
        plt.title('Cumulative proportion vs. PCs Kept')
        plt.xlabel('PCs')
        plt.ylabel('Cumulative Proportion')

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        selected_PCs = self.get_eigenvectors()[:, pcs_to_keep]
        pca_proj = np.dot(self.A, selected_PCs)
        self.A_proj = pca_proj
        return pca_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        (Week 2)

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars). Data projected onto top K PCs then projected back to data space.

        NOTE: If you normalized, remember to rescale the data projected back to the original data space.
        '''
        if self.e_vecs is None or self.A is None:
            raise ValueError('Find PCs first')
        
        top_k_indices = range(top_k)
        pcs_to_project = self.e_vecs[:, top_k_indices]
        centered_data = self.A - np.mean(self.A, axis=0)
        proj_data = np.dot(centered_data, pcs_to_project)
        recon_data = np.dot(proj_data, pcs_to_project.T)

        if self.normalized:
            recon_data = (recon_data * (self.orig_maxs - self.orig_mins)) + self.orig_mins
        
        recon_data += np.mean(self.A, axis=0)
        return recon_data
    #With help from Lucy Barest

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        (Week 2)

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_0 = [0.1, 0.3] and e_1 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.
        '''
        if self.e_vecs is None or len(self.e_vecs) < 2:
            raise ValueError("PCA must be computed with at least 2 principal components before plotting loadings.")
        pc1 = self.e_vecs[:, 0]
        pc2 = self.e_vecs[:, 1]
        num_vars = len(pc1)
        plt.figure(figsize=(8, 8))

        for i in range(num_vars):
            plt.plot([0, pc1[i]], [0, pc2[i]], 'b-')
            plt.annotate(self.vars[i], (pc1[i], pc2[i]), textcoords="offset points", xytext=(5,5), ha='center')
            plt.xlabel("PC1")

        plt.ylabel("PC2")
        plt.title("Loading Plot for Top 2 Principal Components")
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.grid()
        plt.show() 
