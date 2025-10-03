import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import time

def analyze_chem_dataset(csv_file="CMEHL_physics.csv"):
    """
    Analyze the representativeness of a chemical kinetics dataset
    focusing on gradients, reaction rates, and other key metrics.
    """
    print(f"Reading data from {csv_file}...")
    start_time = time.time()
    
    # Load the data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Extract feature columns (all except dataset_type)
    features = [col for col in df.columns if col != 'dataset_type']
    
    # Assuming first column is temperature, rest are species
    temperature_col = features[0]
    species_cols = features[1:]
    
    # Separate by dataset type
    train_df = df[df['dataset_type'] == 'train']
    test_df = df[df['dataset_type'] == 'test']
    val_df = df[df['dataset_type'] == 'validation']
    
    print(f"\n1. Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # 1. Basic Statistics
    print("\n2. Basic Statistics:")
    stats = df[features].describe().T
    # Add coefficient of variation
    stats['cv'] = stats['std'] / stats['mean']
    print(stats[['mean', 'std', 'min', 'max', 'cv']])
    
    # 2. Mass Conservation Check
    print("\n3. Mass Conservation Check:")
    df['mass_sum'] = df[species_cols].sum(axis=1)
    mass_error = np.abs(df['mass_sum'] - 1.0)
    print(f"   Mean absolute error: {mass_error.mean():.6f}")
    print(f"   Max absolute error: {mass_error.max():.6f}")
    print(f"   Mass sum range: [{df['mass_sum'].min():.6f}, {df['mass_sum'].max():.6f}]")
    
    # Sample data if dataset is large
    sample_size = min(5000, len(df))
    if len(df) > sample_size:
        print(f"\nSampling {sample_size} points for plots (from {len(df)} total)")
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # 3. Gradient Analysis
    print("\n4. Gradient Analysis (Rate of Change):")
    
    # Calculate gradients for each dataset type separately
    gradients = {}
    for name, dataset in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        if len(dataset) > 1:
            # Sort the dataset (assuming it has some implicit time ordering)
            dataset_sorted = dataset.sort_index()
            # Calculate gradients efficiently
            dataset_grad = dataset_sorted[features].diff().fillna(0)
            gradients[name] = dataset_grad
    
    # Show gradient statistics for training data
    if 'train' in gradients:
        print("\nGradient statistics for training data:")
        grad_stats = gradients['train'].describe().T
        print(grad_stats[['mean', 'std', 'min', 'max']])
    
    # Plot gradient distributions for key species - LIMIT TO MAX 4
    # Always include temperature and up to 3 important species
    temp_species = [temperature_col]  # Temperature
    chem_species = []
    
    # Add key species in order of priority
    for s in ['H2', 'O2', 'H2O', 'OH']:
        if s in species_cols and len(chem_species) < 3:
            chem_species.append(s)
    
    key_species = temp_species + chem_species
    
    print(f"\nPlotting gradient distributions for {len(key_species)} species...")
    plt.figure(figsize=(12, 8))
    
    for i, species in enumerate(key_species):
        plt.subplot(2, 2, i+1)  # 2x2 grid = 4 plots maximum
        for name, grads in gradients.items():
            if len(grads) > 0:
                # Sample for faster plotting if needed
                if len(grads) > 1000:
                    grads_sample = grads[species].sample(1000, random_state=42)
                else:
                    grads_sample = grads[species]
                    
                sns.histplot(grads_sample, kde=True, label=name, alpha=0.7)
                
        plt.title(f'Gradient of {species}')
        plt.xlabel(f'Rate of Change (d{species}/dt)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Reaction Rate Analysis - ENSURE 4 PLOTS MAX
    print("\n5. Reaction Rate Analysis:")
    
    # Calculate gradients for the sampled data
    df_sample_sorted = df_sample.sort_index()
    all_gradients = df_sample_sorted[features].diff().fillna(0)
    
    # Define groups of species to plot
    plot_groups = [
        # Group 1: Reactants consumption
        {'title': 'Reactant Consumption Rates', 
         'species': ['H2', 'O2'], 
         'label': 'Consumption Rate (-dY/dt)',
         'transform': lambda x: -x},  # Negate to show consumption as positive
         
        # Group 2: Products formation
        {'title': 'Product Formation Rates',
         'species': ['H2O', 'OH'],
         'label': 'Formation Rate (dY/dt)',
         'transform': lambda x: x},
         
        # Group 3: Temperature rate
        {'title': 'Temperature Change Rate',
         'species': [temperature_col],
         'label': 'dT/dt',
         'transform': lambda x: x},
         
        # Group 4: Radical species 
        {'title': 'Radical Species Rates',
         'species': ['H', 'O', 'HO2', 'H2O2'],
         'label': 'Rate (dY/dt)',
         'transform': lambda x: x}
    ]
    
    # Make sure we only include species that exist in the dataset
    for group in plot_groups:
        group['species'] = [s for s in group['species'] if s in features]
    
    # Only keep non-empty groups
    plot_groups = [g for g in plot_groups if g['species']]
    
    # Ensure we have at most 4 groups (for 2x2 grid)
    plot_groups = plot_groups[:4]
    
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(plot_groups):
        plt.subplot(2, 2, i+1)  # 2x2 grid
        
        for species in group['species']:
            # Apply transformation (like negation for consumption)
            y_data = group['transform'](all_gradients[species])
            plt.scatter(df_sample[temperature_col], y_data, 
                      alpha=0.3, label=species, s=10)
                      
        plt.title(group['title'])
        plt.xlabel('Temperature')
        plt.ylabel(group['label'])
        if len(group['species']) > 1:
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Temperature change vs reactant concentrations
    if 'H2' in df.columns and 'O2' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_sample['H2'] * df_sample['O2'], all_gradients[temperature_col], 
                   alpha=0.3, s=10)
        plt.title('Temperature Change Rate vs H2*O2 Product')
        plt.xlabel('H2 * O2 (product of mass fractions)')
        plt.ylabel('dT/dt')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 5. State Space Coverage Analysis using PCA
    print("\n6. State Space Coverage Analysis:")
    
    # PCA for dimensionality reduction
    print("Running PCA...")
    # Use a sample for PCA if dataset is large
    pca_sample = df_sample[features].sample(min(5000, len(df_sample)), random_state=42)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_sample)
    
    # Create a mapping from sample index to dataset type
    sample_indices = pca_sample.index
    pca_dataset_types = df.loc[sample_indices, 'dataset_type'].values
    
    # Plot PCA with dataset types
    plt.figure(figsize=(10, 6))
    for dataset, color, marker in zip(
        ['train', 'validation', 'test'], 
        ['blue', 'green', 'red'],
        ['o', 's', '^']
    ):
        mask = pca_dataset_types == dataset
        if mask.sum() > 0:
            plt.scatter(
                pca_result[mask, 0], 
                pca_result[mask, 1],
                c=color, 
                marker=marker,
                label=dataset,
                alpha=0.5,
                s=20
            )
    
    plt.title('PCA of Chemical State Space')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot PCA colored by temperature
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_result[:, 0], 
        pca_result[:, 1],
        c=pca_sample[temperature_col],
        cmap='plasma',
        alpha=0.5,
        s=20
    )
    plt.colorbar(scatter, label=temperature_col)
    plt.title('Chemical State Space Colored by Temperature')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 6. Species Profile Analysis - use binning for large datasets
    print("\n7. Species Profile Analysis:")
    
    # Sort by temperature and bin data
    print("Analyzing species profiles...")
    
    # Use binning for smoother curves
    temp_bins = np.linspace(df[temperature_col].min(), df[temperature_col].max(), 100)
    bin_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
    
    # Initialize arrays to hold binned species concentrations
    binned_species = {species: np.zeros(len(bin_centers)) for species in species_cols}
    
    # Bin data
    for i in range(len(bin_centers)):
        mask = (df[temperature_col] >= temp_bins[i]) & (df[temperature_col] < temp_bins[i+1])
        if mask.sum() > 0:
            for species in species_cols:
                binned_species[species][i] = df.loc[mask, species].mean()
    
    # Normalize
    for species in species_cols:
        max_val = max(binned_species[species])
        if max_val > 0:
            binned_species[species] = binned_species[species] / max_val
    
    # Plot - divide species into 2 plots if there are many
    if len(species_cols) > 5:
        # Major species
        plt.figure(figsize=(12, 6))
        major_species = ['H2', 'O2', 'H2O', 'N2']
        for species in species_cols:
            if species in major_species or (len([s for s in major_species if s in species_cols]) < 3):
                plt.plot(bin_centers, binned_species[species], label=species)
        
        plt.xlabel(temperature_col)
        plt.ylabel('Normalized Mass Fraction')
        plt.title('Major Species Profiles vs Temperature')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Minor species
        plt.figure(figsize=(12, 6))
        minor_species = [s for s in species_cols if s not in major_species]
        for species in minor_species[:8]:  # Limit to 8 minor species
            plt.plot(bin_centers, binned_species[species], label=species)
        
        plt.xlabel(temperature_col)
        plt.ylabel('Normalized Mass Fraction')
        plt.title('Minor Species Profiles vs Temperature')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # All species in one plot if there aren't many
        plt.figure(figsize=(12, 6))
        for species in species_cols:
            plt.plot(bin_centers, binned_species[species], label=species)
        
        plt.xlabel(temperature_col)
        plt.ylabel('Normalized Mass Fraction')
        plt.title('Normalized Species Profiles vs Temperature')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Find temperature at OH peak (if OH exists)
    if 'OH' in df.columns:
        oh_max_idx = df['OH'].idxmax()
        oh_max_temp = df.loc[oh_max_idx, temperature_col]
        print(f"\nPeak OH condition:")
        print(f"   Temperature at OH peak: {oh_max_temp:.2f}")
        print(f"   Species mass fractions at OH peak:")
        for s in species_cols:
            print(f"      {s}: {df.loc[oh_max_idx, s]:.6f}")
    
    # 7. Estimate reaction rates using approximate stoichiometry
    print("\n8. Estimated Reaction Rates:")
    
    # H2 + 0.5 O2 -> H2O reaction rate (simplified)
    if all(x in df.columns for x in ['H2', 'O2', 'H2O']):
        # Create a basic estimate of reaction rate based on H2 consumption
        df_sample['H2_rxn_rate'] = -all_gradients['H2']
        
        # Plot reaction rate vs temperature
        plt.figure(figsize=(10, 6))
        plt.scatter(df_sample[temperature_col], df_sample['H2_rxn_rate'], alpha=0.3, s=10)
        plt.title('H2 Oxidation Rate vs Temperature')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Approximate Reaction Rate (-dH2/dt)')
        plt.grid(True)
        plt.show()
        
        # Estimate reaction rate constants (very simplified)
        # r = k[H2][O2]^0.5
        df_sample['k_est'] = df_sample['H2_rxn_rate'] / (df_sample['H2'] * np.sqrt(df_sample['O2']))
        df_sample['k_est'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_sample = df_sample.dropna(subset=['k_est'])
        
        if len(df_sample) > 0:
            df_sample['log_k'] = np.log(df_sample['k_est'].abs())
            df_sample['inv_T'] = 1 / df_sample[temperature_col]
            
            # Plot Arrhenius behavior (log k vs 1/T)
            plt.figure(figsize=(10, 6))
            plt.scatter(df_sample['inv_T'], df_sample['log_k'], alpha=0.3, s=10)
            plt.title('Arrhenius Plot (ln(k) vs 1/T)')
            plt.xlabel('1/T (K^-1)')
            plt.ylabel('ln(k)')
            plt.grid(True)
            plt.show()
    
    end_time = time.time()
    print(f"\nAnalysis complete in {end_time - start_time:.2f} seconds.")
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "CMEHL_physics.csv"
    
    try:
        analyze_chem_dataset(csv_file)
    except Exception as e:
        print(f"Error: {e}")
        