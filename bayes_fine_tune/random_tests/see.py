import matplotlib.pyplot as plt
import numpy as np
import sys
# Read and parse the data
methods = []
correlations = []
mses = []
mutations = []

results_file = sys.argv[1]

with open(results_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 4:  # Changed from 3 to 4 due to noise field
            continue
            
        # Extract M value (number of mutations)
        m = int(parts[2].split('=')[1])  # Changed from parts[1] to parts[2]
        mutations.append(m)
        
        # Extract method names and their scores
        current_methods = []
        current_corrs = []
        current_mses = []
        
        for i in range(6, len(parts), 3):
            if i + 2 >= len(parts):
                break
            method = parts[i]
            corr = float(parts[i+1])
            mse = float(parts[i+2])
            
            current_methods.append(method)
            current_corrs.append(corr)
            current_mses.append(mse)
            
        if not methods:  # First line
            methods = current_methods
        correlations.append(current_corrs)
        mses.append(current_mses)

# Convert to numpy arrays for easier plotting
correlations = np.array(correlations)
mses = np.array(mses)
mutations = np.array(mutations)

methods_subset = methods
#methods_subset = ['masked_marginal', 'path_chaining', 'wildtype_marginal']

# Create the correlation plot
plt.figure(figsize=(12, 6))
for i, method in enumerate(methods):
    if method not in methods_subset:
        continue
    plt.plot(mutations, correlations[:, i], marker='o', label=method)
plt.xlabel('Number of Mutations (M)')
plt.ylabel('Correlation')
plt.title('Correlation vs Number of Mutations')
plt.grid(True)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("images/" + results_file.replace('.out', '_correlation_plot.png'), bbox_inches='tight', dpi=300)
plt.close()

# Create the MSE plot
plt.figure(figsize=(12, 6))
for i, method in enumerate(methods):
    plt.plot(mutations, mses[:, i], marker='o', label=method)
plt.xlabel('Number of Mutations (M)')
plt.ylabel('MSE')
plt.title('MSE vs Number of Mutations (noise=1.0)')
plt.grid(True)
plt.yscale('log')  # Use log scale for MSE since values vary greatly
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("images/" + results_file.replace('.out', '_mse_plot.png'), bbox_inches='tight', dpi=300)
plt.close()
