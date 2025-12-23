"""
Main script to run the complete GMM pipeline
"""

from gmm import *
import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("GAUSSIAN MIXTURE MODEL - M.Sc. PROJECT")
    print("=" * 60)
    
    # Create output directories
    create_output_directories()
    
    # ===== STEP 1: Load and analyze data =====
    print("\n[STEP 1] LOADING AND ANALYZING DATA")
    print("-" * 40)
    
    # Load your data
    data_path = "data/dataset1.csv"  # Change this to your actual file
    
    # Try to load data
    df = DataProcessor.load_data(data_path)
    
    if df is None:
        print("\nCreating sample data for testing...")
        np.random.seed(42)
        n_samples = 300
        
        # Create 3 Gaussian clusters in 3D for better visualization
        data1 = np.random.multivariate_normal([0, 0, 0], [[1, 0.7, 0.3], [0.7, 1, 0.4], [0.3, 0.4, 1]], n_samples//3)
        data2 = np.random.multivariate_normal([5, 5, 5], [[1, -0.5, 0.2], [-0.5, 1, -0.3], [0.2, -0.3, 1]], n_samples//3)
        data3 = np.random.multivariate_normal([-4, 3, 2], [[0.5, 0, 0.1], [0, 2, 0], [0.1, 0, 0.8]], n_samples//3)
        
        X = np.vstack([data1, data2, data3])
        df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
        
        # Save sample data
        df.to_csv("data/dataset1.csv", index=False)
        print("✓ Sample data created and saved to data/sample_data.csv")
        print(f"  Shape: {df.shape}")
    
    # Analyze data structure
    numeric_cols, text_cols = DataProcessor.analyze_data(df)
    
    if len(numeric_cols) == 0:
        print("\n✗ ERROR: No numeric columns found in the dataset!")
        print("Please check your dataset format. The GMM algorithm requires numeric data.")
        return
    
    # ===== STEP 2: Visualize raw data (BEFORE processing) =====
    print("\n[STEP 2] DATA VISUALIZATION")
    print("-" * 40)
    
    # Create numeric dataframe for visualization
    df_numeric = df[numeric_cols].copy()
    
    # Visualize the raw numeric data
    DataProcessor.visualize_data(df_numeric)
    
    # ===== STEP 3: Preprocess data =====
    print("\n[STEP 3] DATA PREPROCESSING")
    print("-" * 40)
    
    # Clean and normalize numeric data
    df_cleaned = DataProcessor.clean_data(df, numeric_cols)
    df_normalized = DataProcessor.normalize_data(df_cleaned)
    
    # Convert to numpy array for GMM
    X = df_normalized.values
    print(f"✓ Data ready for GMM: {X.shape}")
    
    # ===== STEP 4: Determine optimal number of clusters =====
    print("\n[STEP 4] DETERMINING OPTIMAL CLUSTERS")
    print("-" * 40)
    
    # Simple heuristic for number of clusters
    if X.shape[0] < 50:
        n_components = 2
    elif X.shape[0] < 100:
        n_components = 3
    elif X.shape[0] < 200:
        n_components = min(5, X.shape[0] // 20)
    else:
        n_components = min(10, X.shape[0] // 30)
    
    print(f"Suggested number of clusters: {n_components}")
    print("(Based on sample size and common practice)")
    
    # Allow user to override
    try:
        user_input = input(f"Press Enter to use {n_components} clusters, or enter a different number: ").strip()
        if user_input and user_input.isdigit():
            n_components = int(user_input)
            print(f"✓ Using {n_components} clusters as specified")
    except:
        print(f"✓ Using default: {n_components} clusters")
    
    # ===== STEP 5: Train GMM =====
    print("\n[STEP 5] TRAINING GMM")
    print("-" * 40)
    
    # Create and train GMM
    gmm = GaussianMixtureModel(
        n_components=n_components,
        max_iter=200,
        tol=1e-4,
        random_state=42
    )
    
    gmm.fit(X, init_method='kmeans', verbose=True)
    
    # ===== STEP 6: Make predictions =====
    print("\n[STEP 6] MAKING PREDICTIONS")
    print("-" * 40)
    
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"✓ Predicted clusters: {len(unique_labels)}")
    print(f"✓ Cluster distribution:")
    for label, count in zip(unique_labels, label_counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {label}: {count} samples ({percentage:.1f}%)")
    
    # ===== STEP 7: Visualize results =====
    print("\n[STEP 7] VISUALIZING RESULTS")
    print("-" * 40)
    
    # Reduce dimensionality for visualization if needed
    if X.shape[1] > 2:
        print("Reducing dimensionality for visualization using PCA...")
        X_vis = reduce_dimensionality(X, n_components=2)
    else:
        X_vis = X
    
    # Plot clusters
    print("Generating cluster visualizations...")
    GMMVisualizer.plot_clusters(
        X_vis, labels, gmm.means,
        save_path="outputs/plots/clusters.png"
    )
    
    # Plot convergence
    if gmm.log_likelihoods:
        GMMVisualizer.plot_convergence(
            gmm.log_likelihoods,
            save_path="outputs/plots/convergence.png"
        )
    
    # Plot cluster statistics
    GMMVisualizer.plot_cluster_stats(
        labels,
        save_path="outputs/plots/cluster_stats.png"
    )
    
    # ===== STEP 8: Evaluate model =====
    print("\n[STEP 8] MODEL EVALUATION")
    print("-" * 40)
    
    # Compute evaluation metrics
    silhouette = GMMEvaluator.silhouette_score(X, labels)
    bic = GMMEvaluator.BIC_score(X, gmm)
    aic = GMMEvaluator.AIC_score(X, gmm)
    
    print(f"✓ Silhouette Score: {silhouette:.4f}")
    print(f"✓ BIC Score: {bic:.4f}")
    print(f"✓ AIC Score: {aic:.4f}")
    
    # Interpret silhouette score
    print(f"\nSilhouette Score Interpretation:")
    if silhouette > 0.7:
        print("  Excellent cluster separation")
    elif silhouette > 0.5:
        print("  Reasonable cluster structure")
    elif silhouette > 0.25:
        print("  Weak cluster structure")
    else:
        print("  No substantial cluster structure")
    
    # ===== STEP 9: Generate synthetic samples =====
    print("\n[STEP 9] GENERATING SYNTHETIC SAMPLES")
    print("-" * 40)
    
    synthetic_samples = gmm.sample(n_samples=50)
    print(f"✓ Generated {len(synthetic_samples)} synthetic samples")
    
    # Save synthetic samples
    synthetic_df = pd.DataFrame(synthetic_samples, 
                               columns=[f'Feature_{i}' for i in range(synthetic_samples.shape[1])])
    synthetic_df.to_csv("outputs/results/synthetic_samples.csv", index=False)
    print("✓ Synthetic samples saved to outputs/results/synthetic_samples.csv")
    
    # ===== STEP 10: Save results =====
    print("\n[STEP 10] SAVING RESULTS")
    print("-" * 40)
    
    # Generate and save report
    ReportGenerator.generate_summary(gmm, X, labels, 
                                   file_path="outputs/results/gmm_report.txt")
    
    # Save model
    ReportGenerator.save_model(gmm, "outputs/models/gmm_model.pkl")
    
    # Save predictions back to CSV with cluster labels
    results_df = df.copy()
    results_df['GMM_Cluster'] = labels
    
    # Add cluster probabilities for the top clusters
    n_proba_cols = min(5, n_components)
    for k in range(n_proba_cols):
        if k < probabilities.shape[1]:
            results_df[f'Cluster_{k}_Probability'] = probabilities[:, k]
    
    results_df.to_csv("outputs/results/clustered_results.csv", index=False)
    print("✓ Results saved to outputs/results/clustered_results.csv")
    
    # Save cluster statistics
    cluster_stats = pd.DataFrame({
        'Cluster': unique_labels,
        'Count': label_counts,
        'Percentage': (label_counts / len(labels)) * 100
    })
    cluster_stats.to_csv("outputs/results/cluster_statistics.csv", index=False)
    print("✓ Cluster statistics saved to outputs/results/cluster_statistics.csv")
    
    print("\n" + "=" * 60)
    print("🎉 PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Print final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Clusters: {n_components} (found {len(unique_labels)} non-empty)")
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"   Model converged: {gmm.converged}")
    print(f"   Iterations: {len(gmm.log_likelihoods)}")
    
    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"   1. Plots in 'outputs/plots/' folder:")
    print(f"      - clusters.png (Cluster visualization)")
    print(f"      - convergence.png (EM convergence)")
    print(f"      - cluster_stats.png (Cluster statistics)")
    print(f"      - data_distributions.png (Data analysis)")
    print(f"      - correlation_heatmap.png (Feature correlations)")
    print(f"      - box_plots.png (Feature distributions)")
    
    print(f"\n   2. Results in 'outputs/results/' folder:")
    print(f"      - gmm_report.txt (Detailed report)")
    print(f"      - clustered_results.csv (Data with clusters)")
    print(f"      - cluster_statistics.csv (Cluster stats)")
    print(f"      - synthetic_samples.csv (Generated data)")
    
    print(f"\n   3. Model in 'outputs/models/' folder:")
    print(f"      - gmm_model.pkl (Trained GMM model)")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Check the generated plots in 'outputs/plots/'")
    print(f"   2. Read 'outputs/results/gmm_report.txt' for analysis")
    print(f"   3. Try different cluster counts for better results")
    print(f"   4. Analyze your data patterns in clustered_results.csv")

if __name__ == "__main__":
    main()