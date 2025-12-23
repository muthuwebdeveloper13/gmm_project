"""
COMPLETE Gaussian Mixture Model Implementation
All modules in one file - Easy to debug
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import os

# ==================== MODULE 1: Data Preprocessing ====================
class DataProcessor:
    """Load, clean, and normalize data"""
    
    @staticmethod
    def load_data(file_path):
        """Load dataset from CSV"""
        print(f"Loading data from: {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Data loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    @staticmethod
    def analyze_data(df):
        """Analyze the dataset structure"""
        print("\n" + "="*50)
        print("DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"Total samples: {df.shape[0]}")
        print(f"Total features: {df.shape[1]}")
        
        # Column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        print(f"\nNumeric columns ({len(numeric_cols)}):")
        if len(numeric_cols) > 0:
            print(f"  {list(numeric_cols)}")
        else:
            print("  No numeric columns found!")
        
        print(f"\nText columns ({len(text_cols)}):")
        if len(text_cols) > 0:
            print(f"  {list(text_cols[:5])}{'...' if len(text_cols) > 5 else ''}")
        
        # Basic statistics for numeric columns
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics for numeric columns:")
            stats_df = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
            print(stats_df.head(10))
        
        # Missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print(f"\nMissing values:")
            for col in missing_cols.index:
                percent = (missing_cols[col] / len(df)) * 100
                print(f"  {col}: {missing_cols[col]} ({percent:.1f}%)")
        else:
            print("\n✓ No missing values found")
        
        return numeric_cols, text_cols
    
    @staticmethod
    def clean_data(df, numeric_cols=None):
        """Handle missing values and outliers - ONLY for numeric columns"""
        print("\nCleaning data...")
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("  No numeric columns to clean")
            return df
        
        # Create a copy for numeric data
        df_numeric = df[numeric_cols].copy()
        
        # 1. Handle missing values in numeric columns only
        missing_before = df_numeric.isnull().sum().sum()
        
        if missing_before > 0:
            # Fill numeric missing values with column mean
            for col in numeric_cols:
                if df_numeric[col].isnull().any():
                    mean_val = df_numeric[col].mean()
                    df_numeric[col] = df_numeric[col].fillna(mean_val)
            print(f"  Filled {missing_before} missing values")
        else:
            print(f"  No missing values to fill")
        
        # 2. Remove extreme outliers (optional)
        outliers_capped = 0
        for col in numeric_cols:
            Q1 = df_numeric[col].quantile(0.25)
            Q3 = df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Only cap if there's variability
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count and cap outliers
                outliers = ((df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)).sum()
                outliers_capped += outliers
                
                df_numeric[col] = np.where(df_numeric[col] < lower_bound, lower_bound, df_numeric[col])
                df_numeric[col] = np.where(df_numeric[col] > upper_bound, upper_bound, df_numeric[col])
        
        if outliers_capped > 0:
            print(f"  Capped {outliers_capped} outliers")
        else:
            print(f"  No outliers to cap")
        
        return df_numeric
    
    @staticmethod
    def normalize_data(df_numeric):
        """Normalize numeric data using Z-score"""
        print("\nNormalizing numeric data...")
        
        if df_numeric.empty:
            print("  No numeric data to normalize")
            return df_numeric
        
        # Store original for reference
        df_normalized = df_numeric.copy()
        
        normalized_count = 0
        for col in df_numeric.columns:
            std = df_numeric[col].std()
            
            if std > 0:  # Avoid division by zero
                mean = df_numeric[col].mean()
                df_normalized[col] = (df_numeric[col] - mean) / std
                normalized_count += 1
            else:
                print(f"  '{col}': Constant value, skipping normalization")
        
        print(f"  Normalized {normalized_count} columns using Z-score")
        return df_normalized
    
    @staticmethod
    def visualize_data(df_numeric):
        """Plot numeric data distribution with better visualizations"""
        print("\nVisualizing data...")
        
        if df_numeric.empty:
            print("  No numeric data to visualize")
            return
        
        n_features = df_numeric.shape[1]
        
        # If too many features, show only the first 15
        if n_features > 15:
            print(f"  Showing first 15 features out of {n_features}")
            df_to_plot = df_numeric.iloc[:, :15]
        else:
            df_to_plot = df_numeric
        
        features_to_plot = df_to_plot.columns.tolist()
        n_plots = len(features_to_plot)
        
        # Create figure with subplots
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Set figure size based on number of rows
        fig_height = max(4 * n_rows, 6)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, fig_height))
        
        # Flatten axes array for easy indexing
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Create plots for each feature
        for idx, col in enumerate(features_to_plot):
            ax = axes[idx]
            
            # Create histogram
            values = df_to_plot[col].dropna()
            n_bins = min(30, len(values) // 10)
            n_bins = max(5, n_bins)  # Ensure at least 5 bins
            
            # Histogram
            n, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                      color='steelblue', edgecolor='black')
            
            # Add vertical line for mean
            mean_val = values.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}')
            
            # Add vertical lines for ±1 std
            std_val = values.std()
            if std_val > 0:
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                          linewidth=1.5, alpha=0.7, label=f'±1σ: {std_val:.2f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', 
                          linewidth=1.5, alpha=0.7)
            
            # Add KDE curve
            from scipy.stats import gaussian_kde
            if len(values) > 1:
                try:
                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    kde_vals = kde(x_range)
                    # Scale KDE to match histogram
                    kde_scale = n.max() / kde_vals.max()
                    ax.plot(x_range, kde_vals * kde_scale, 'g-', linewidth=2, 
                           alpha=0.7, label='KDE')
                except:
                    pass
            
            # Customize subplot
            ax.set_title(f'{col}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8, loc='upper right')
            
            # Add statistics text box
            stats_text = f'n={len(values)}\nμ={mean_val:.2f}\nσ={std_val:.2f}\nmin={values.min():.2f}\nmax={values.max():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions with Statistics', fontsize=14, 
                    fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create correlation heatmap if we have multiple features
        if len(features_to_plot) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = df_to_plot.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            cmap = plt.cm.RdYlBu_r
            im = plt.imshow(corr_matrix, cmap=cmap, aspect='auto', 
                           vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(features_to_plot)):
                for j in range(len(features_to_plot)):
                    if not mask[i, j]:  # Only show lower triangle
                        text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha='center', va='center', 
                                       color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white',
                                       fontsize=8)
            
            plt.colorbar(im, label='Correlation Coefficient', fraction=0.046, pad=0.04)
            plt.xticks(range(len(features_to_plot)), features_to_plot, 
                      rotation=45, ha='right', fontsize=9)
            plt.yticks(range(len(features_to_plot)), features_to_plot, fontsize=9)
            plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Create box plots for key features
        if len(features_to_plot) > 1:
            # Select up to 10 features for box plot
            n_box_features = min(10, len(features_to_plot))
            box_features = features_to_plot[:n_box_features]
            
            plt.figure(figsize=(12, 6))
            box_data = [df_to_plot[col].dropna() for col in box_features]
            
            bp = plt.boxplot(box_data, labels=box_features, patch_artist=True)
            
            # Customize box colors
            colors = plt.cm.Set3(np.linspace(0, 1, n_box_features))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize whiskers and medians
            for whisker in bp['whiskers']:
                whisker.set(color='gray', linewidth=1.5, linestyle='--')
            
            for median in bp['medians']:
                median.set(color='red', linewidth=2)
            
            plt.ylabel('Values', fontsize=11)
            plt.title('Box Plots of Features', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
            plt.show()

# ==================== MAIN GMM CLASS ====================
class GaussianMixtureModel:
    """Complete GMM implementation with EM algorithm"""
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Model parameters
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None
        
        # Training history
        self.log_likelihoods = []
        self.converged = False
    
    def _compute_pdf(self, X, mean, cov):
        """Compute Gaussian PDF for a single component"""
        n_features = X.shape[1]
        
        # Add regularization for numerical stability
        cov_reg = cov + np.eye(n_features) * 1e-6
        
        try:
            return multivariate_normal(mean=mean, cov=cov_reg).pdf(X)
        except:
            # Manual calculation if scipy fails
            X_centered = X - mean
            try:
                cov_inv = np.linalg.inv(cov_reg)
            except:
                # If matrix is singular, use pseudo-inverse
                cov_inv = np.linalg.pinv(cov_reg)
            
            exponent = -0.5 * np.sum(X_centered @ cov_inv * X_centered, axis=1)
            norm_const = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(np.linalg.det(cov_reg)))
            return norm_const * np.exp(exponent)
    
    def _e_step(self, X):
        """Expectation step - compute responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute weighted PDFs for each component
        for k in range(self.n_components):
            pdf = self._compute_pdf(X, self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * pdf
        
        # Normalize responsibilities
        sum_resp = responsibilities.sum(axis=1, keepdims=True)
        sum_resp[sum_resp == 0] = 1e-15  # Avoid division by zero
        responsibilities = responsibilities / sum_resp
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """Maximization step - update parameters"""
        n_samples, n_features = X.shape
        
        # Update weights
        new_weights = responsibilities.sum(axis=0) / n_samples
        
        # Update means
        new_means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            sum_resp_k = resp_k.sum()
            
            if sum_resp_k > 1e-10:  # Avoid division by very small numbers
                new_means[k] = np.sum(X * resp_k[:, np.newaxis], axis=0) / sum_resp_k
            else:
                # If component has no responsibility, reinitialize
                new_means[k] = X[np.random.randint(0, n_samples)]
        
        # Update covariances
        new_covariances = []
        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            sum_resp_k = resp_k.sum()
            
            if sum_resp_k > 1e-10:
                X_centered = X - new_means[k]
                weighted_cov = (resp_k[:, np.newaxis] * X_centered).T @ X_centered
                cov = weighted_cov / sum_resp_k
                # Add regularization
                cov = cov + np.eye(n_features) * 1e-6
                new_covariances.append(cov)
            else:
                # Reinitialize covariance
                new_covariances.append(np.eye(n_features))
        
        return new_means, np.array(new_covariances), new_weights
    
    def _compute_log_likelihood(self, X):
        """Compute total log-likelihood"""
        total = 0
        for k in range(self.n_components):
            pdf = self._compute_pdf(X, self.means[k], self.covariances[k])
            total += np.log(self.weights[k] * pdf + 1e-15).sum()
        return total
    
    def initialize_parameters(self, X, method='kmeans'):
        """Initialize GMM parameters"""
        n_samples, n_features = X.shape
        
        if method == 'kmeans':
            # Use K-means for initialization
            kmeans = KMeans(n_clusters=self.n_components, 
                          random_state=self.random_state,
                          n_init=10)
            kmeans.fit(X)
            self.means = kmeans.cluster_centers_
            
            # Initialize covariances based on cluster assignments
            self.covariances = []
            labels = kmeans.labels_
            for k in range(self.n_components):
                cluster_points = X[labels == k]
                if len(cluster_points) > 1:
                    cov = np.cov(cluster_points.T)
                    # Regularize
                    cov = cov + np.eye(n_features) * 1e-6
                else:
                    cov = np.eye(n_features)
                self.covariances.append(cov)
            self.covariances = np.array(self.covariances)
            
        else:  # Random initialization
            # Random means
            idx = np.random.choice(n_samples, self.n_components, replace=False)
            self.means = X[idx].copy()
            
            # Random covariances
            self.covariances = np.array([np.eye(n_features) 
                                       for _ in range(self.n_components)])
        
        # Equal weights
        self.weights = np.ones(self.n_components) / self.n_components
    
    def fit(self, X, init_method='kmeans', verbose=True):
        """Train GMM using EM algorithm"""
        n_samples, n_features = X.shape
        
        print(f"\nTraining GMM with {self.n_components} components...")
        print(f"Data shape: {X.shape}")
        
        # Initialize parameters
        self.initialize_parameters(X, init_method)
        
        # EM Algorithm
        for iteration in range(self.max_iter):
            # E-Step
            self.responsibilities = self._e_step(X)
            
            # M-Step
            self.means, self.covariances, self.weights = self._m_step(X, self.responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                improvement = abs(log_likelihood - self.log_likelihoods[-2])
                
                if improvement < self.tol:
                    self.converged = True
                    if verbose:
                        print(f"✓ Converged at iteration {iteration + 1}")
                    break
            
            # Print progress
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.4f}")
        
        if not self.converged:
            print(f"  Max iterations ({self.max_iter}) reached")
        
        print(f"  Final log-likelihood: {self.log_likelihoods[-1]:.4f}")
        print(f"  Total iterations: {len(self.log_likelihoods)}")
        
        return self
    
    def predict(self, X):
        """Assign cluster labels"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """Get cluster probabilities"""
        return self._e_step(X)
    
    def sample(self, n_samples=100):
        """Generate new samples"""
        n_features = self.means.shape[1]
        
        # Choose components based on weights
        components = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        
        # Generate samples
        samples = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            comp = components[i]
            samples[i] = np.random.multivariate_normal(
                self.means[comp], 
                self.covariances[comp]
            )
        
        return samples

# ==================== VISUALIZATION ====================
class GMMVisualizer:
    """Visualization functions"""
    
    @staticmethod
    def plot_clusters(X, labels, means, save_path=None):
        """Plot clustered data"""
        n_clusters = len(np.unique(labels))
        
        # Choose color map based on number of clusters
        if n_clusters <= 10:
            cmap = plt.cm.tab10
        elif n_clusters <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.viridis
        
        # Create figure
        if X.shape[1] == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Scatter plot
            scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels, 
                                  cmap=cmap, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            ax1.scatter(means[:, 0], means[:, 1], c='red', marker='X', 
                       s=250, label='Cluster Centers', edgecolor='black', linewidth=2, zorder=5)
            ax1.set_xlabel('Feature 1', fontsize=11)
            ax1.set_ylabel('Feature 2', fontsize=11)
            ax1.set_title('2D Cluster Visualization', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(scatter1, ax=ax1)
            cbar.set_label('Cluster', fontsize=11)
            
            # Plot 2: Density plot
            from scipy.stats import gaussian_kde
            
            # Sort by density
            xy = np.vstack([X[:, 0], X[:, 1]])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            
            scatter2 = ax2.scatter(X[idx, 0], X[idx, 1], c=z[idx], 
                                  cmap='viridis', alpha=0.7, s=50)
            ax2.scatter(means[:, 0], means[:, 1], c='red', marker='X', 
                       s=250, label='Cluster Centers', edgecolor='black', linewidth=2, zorder=5)
            ax2.set_xlabel('Feature 1', fontsize=11)
            ax2.set_ylabel('Feature 2', fontsize=11)
            ax2.set_title('Density-based Visualization', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            plt.colorbar(scatter2, ax=ax2, label='Density')
            
        else:
            # For high-dimensional data, use PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            means_pca = pca.transform(means) if means is not None else None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: PCA scatter plot
            scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                  cmap=cmap, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            if means_pca is not None:
                ax1.scatter(means_pca[:, 0], means_pca[:, 1], c='red', marker='X', 
                           s=250, label='Cluster Centers', edgecolor='black', linewidth=2, zorder=5)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
            ax1.set_title('PCA-based Cluster Visualization', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            plt.colorbar(scatter1, ax=ax1, label='Cluster')
            
            # Plot 2: Explained variance
            ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Principal Component', fontsize=11)
            ax2.set_ylabel('Explained Variance (%)', fontsize=11)
            ax2.set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_convergence(log_likelihoods, save_path=None):
        """Plot convergence of EM algorithm"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, 
                'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Log-Likelihood', fontsize=12)
        plt.title('EM Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Mark convergence point
        if len(log_likelihoods) > 1:
            for i in range(1, len(log_likelihoods)):
                improvement = abs(log_likelihoods[i] - log_likelihoods[i-1])
                if improvement < 1e-4 or i == len(log_likelihoods) - 1:
                    plt.axvline(x=i+1, color='r', linestyle='--', alpha=0.7,
                              label=f'Final iteration: {i+1}')
                    break
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_cluster_stats(labels, save_path=None):
        """Plot cluster statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique)
        
        # Choose colors
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.arange(n_clusters) / n_clusters)
        elif n_clusters <= 20:
            colors = plt.cm.tab20(np.arange(n_clusters) / n_clusters)
        else:
            colors = plt.cm.viridis(np.arange(n_clusters) / n_clusters)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        bars = ax1.bar(range(n_clusters), counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Cluster', fontsize=11)
        ax1.set_ylabel('Number of Samples', fontsize=11)
        ax1.set_title('Cluster Sizes', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        explode = [0.05] * n_clusters
        wedges, texts, autotexts = ax2.pie(counts, 
                                          labels=[f'Cluster {i}' for i in range(n_clusters)],
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=colors,
                                          explode=explode,
                                          shadow=True)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('Cluster Distribution', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ==================== EVALUATION ====================
class GMMEvaluator:
    """Evaluation metrics for GMM"""
    
    @staticmethod
    def silhouette_score(X, labels):
        """Compute silhouette score"""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return -1
        return silhouette_score(X, labels)
    
    @staticmethod
    def BIC_score(X, gmm):
        """Compute Bayesian Information Criterion"""
        n_samples, n_features = X.shape
        k = gmm.n_components
        
        # Number of parameters
        n_params = (k * n_features + 
                   k * n_features * (n_features + 1) / 2 + 
                   k - 1)
        
        # Compute log-likelihood
        log_likelihood = gmm.log_likelihoods[-1] if gmm.log_likelihoods else 0
        
        # BIC formula
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        return bic
    
    @staticmethod
    def AIC_score(X, gmm):
        """Compute Akaike Information Criterion"""
        n_samples, n_features = X.shape
        k = gmm.n_components
        
        # Number of parameters
        n_params = (k * n_features + 
                   k * n_features * (n_features + 1) / 2 + 
                   k - 1)
        
        # Compute log-likelihood
        log_likelihood = gmm.log_likelihoods[-1] if gmm.log_likelihoods else 0
        
        # AIC formula
        aic = 2 * n_params - 2 * log_likelihood
        return aic

# ==================== REPORT GENERATION ====================
class ReportGenerator:
    """Generate project report"""
    
    @staticmethod
    def generate_summary(gmm, X, labels, file_path=None):
        """Generate summary report"""
        from datetime import datetime
        
        # Calculate number of clusters from labels
        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels)
        n_clusters_model = gmm.n_components
        
        summary = f"""
{'='*60}
GAUSSIAN MIXTURE MODEL - PROJECT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

1. DATASET INFORMATION
{'-'*30}
- Number of samples: {X.shape[0]:,}
- Number of features: {X.shape[1]}
- Data type: {X.dtype}

2. GMM MODEL DETAILS
{'-'*30}
- Number of components: {n_clusters_model}
- Converged: {gmm.converged}
- Total iterations: {len(gmm.log_likelihoods)}
- Final log-likelihood: {gmm.log_likelihoods[-1] if gmm.log_likelihoods else 'N/A':.4f}
- Tolerance: {gmm.tol}
- Random seed: {gmm.random_state}

3. CLUSTER DISTRIBUTION
{'-'*30}"""
        
        # Add cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        
        for cluster, count, perc in zip(unique, counts, percentages):
            summary += f"\n- Cluster {cluster}: {count} samples ({perc:.1f}%)"
        
        summary += f"\n\n4. EVALUATION METRICS\n{'-'*30}"
        
        # Compute metrics
        silhouette = GMMEvaluator.silhouette_score(X, labels)
        bic = GMMEvaluator.BIC_score(X, gmm)
        aic = GMMEvaluator.AIC_score(X, gmm)
        
        summary += f"""
- Silhouette Score: {silhouette:.4f}
- BIC Score: {bic:.4f}
- AIC Score: {aic:.4f}"""
        
        # Interpretation
        summary += f"\n\n5. INTERPRETATION\n{'-'*30}"
        
        if silhouette > 0.7:
            summary += "\n- Silhouette score > 0.7: Excellent cluster separation"
        elif silhouette > 0.5:
            summary += "\n- Silhouette score > 0.5: Reasonable cluster structure"
        elif silhouette > 0.25:
            summary += "\n- Silhouette score > 0.25: Weak cluster structure"
        else:
            summary += "\n- Silhouette score <= 0.25: No substantial cluster structure"
        
        # Recommendations
        summary += f"\n\n6. RECOMMENDATIONS\n{'-'*30}"
        
        if n_clusters_found < 3:
            summary += "\n- Consider increasing number of clusters for better insights"
        elif n_clusters_found > 10:
            summary += "\n- Consider reducing number of clusters for interpretability"
        
        if silhouette < 0.25:
            summary += "\n- Data may not have clear cluster structure"
            summary += "\n- Consider alternative clustering methods or feature engineering"
        
        if n_clusters_model != n_clusters_found:
            summary += f"\n- Note: Model configured for {n_clusters_model} clusters but found {n_clusters_found} non-empty clusters"
        
        summary += f"\n\n{'='*60}\n"
        
        # Save to file
        if file_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Try UTF-8 encoding first
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    print(f"✓ Report saved to {file_path}")
                except:
                    # Fallback to ASCII
                    with open(file_path, 'w', encoding='ascii', errors='ignore') as f:
                        f.write(summary)
                    print(f"✓ Report saved with ASCII encoding to {file_path}")
            except Exception as e:
                print(f"✗ Could not save report: {e}")
        
        print(summary)
        return summary
    
    @staticmethod
    def save_model(gmm, file_path):
        """Save model parameters"""
        import pickle
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            model_data = {
                'means': gmm.means,
                'covariances': gmm.covariances,
                'weights': gmm.weights,
                'n_components': gmm.n_components,
                'log_likelihoods': gmm.log_likelihoods,
                'converged': gmm.converged,
                'random_state': gmm.random_state
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {file_path}")
            return model_data
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            return None

# ==================== UTILITY FUNCTIONS ====================
def reduce_dimensionality(X, n_components=2):
    """Reduce dimensionality for visualization"""
    from sklearn.decomposition import PCA
    
    if X.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        print(f"PCA applied: {X.shape[1]} features → {n_components} features")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        return X_reduced
    else:
        return X

def create_output_directories():
    """Create output directories if they don't exist"""
    directories = ['outputs', 'outputs/plots', 'outputs/models', 'outputs/results', 'data']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Output directories created")