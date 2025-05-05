import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set larger font sizes
plt.rcParams.update({
    'font.size': 16,           # Default font size
    'axes.labelsize': 20,      # x and y labels
    'xtick.labelsize': 16,     # x-axis ticks
    'ytick.labelsize': 16,     # y-axis ticks
    'legend.fontsize': 16,     # Legend font size
    'axes.titlesize': 20       # Title font size
})

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Read the data
df_recurrent = pd.read_csv("csv/recurrent_merged.csv")
df_parallel = pd.read_csv("csv/parallel_merged.csv")

# Function to create scatter plot for all groups
def plot_all_groups(df, title_prefix):
    plt.figure(figsize=(12, 8))
    
    # Get unique groups
    groups = df.groupby(['batch', 'length', 'heads', 'dim'])
    
    # Create a color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    
    # Plot each group with a different color
    for (batch, length, heads, dim), group in groups:
        group_label = f'B{batch}L{length}H{heads}D{dim}'
        plt.scatter(group['SRAM-size'], group['arithmetic-intensity'],
                   color=colors[list(groups.groups.keys()).index((batch, length, heads, dim))],
                   label=group_label)
        
        # Track annotated points and their positions
        annotated_points = []
        annotation_positions = []
        
        # Try to annotate each point
        for i, (_, row) in enumerate(group.iterrows()):
            point = (row['SRAM-size'], row['arithmetic-intensity'])
            annotation_pos = (point[0] + 5, point[1] + 5)  # Default offset
            
            # Check if this annotation would overlap with any existing ones
            overlaps = False
            for existing_pos in annotation_positions:
                # Calculate distance between annotation positions
                dist = np.sqrt((annotation_pos[0] - existing_pos[0])**2 + 
                              (annotation_pos[1] - existing_pos[1])**2)
                if dist < 10:  # Minimum distance between annotations
                    overlaps = True
                    break
            
            if not overlaps:
                plt.annotate(f'({row["BK"]},{row["BV"]},{row["BT"]})', 
                            point,
                            xytext=(5, 5), 
                            textcoords='offset points',
                            fontsize=8)
                annotated_points.append(i)
                annotation_positions.append(annotation_pos)


    # Add labels and title
    plt.xlabel('SRAM Size')
    plt.ylabel('Arithmetic Intensity')
    plt.title(f'{title_prefix}\nSRAM Size vs Arithmetic Intensity')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend inside the plot at bottom right
    plt.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot to plots directory
    plt.savefig(f'plots/analysis_{title_prefix.lower().replace(" ", "_")}_sram_vs_ai.pdf', 
                bbox_inches='tight', dpi=300)
    plt.close()

def make_roofline_plot():
    # Hardware parameters for NVIDIA A100 (FP32)
    peak_flops = 19.5e12  # 19.5 TFLOPS
    peak_bandwidth = 1.555e12 * 2# 1555 GB/s

    # Get unique groups from either dataframe (they should have the same groups)
    groups = df_parallel.groupby(['batch', 'length', 'heads', 'dim'])
    
    # Create one plot per group
    for (batch, length, heads, dim), _ in groups:
        group_label = f'B{batch}L{length}H{heads}D{dim}'
        
        # Get the data for this group from both dataframes
        parallel_group = df_parallel[
            (df_parallel['batch'] == batch) & 
            (df_parallel['length'] == length) & 
            (df_parallel['heads'] == heads) & 
            (df_parallel['dim'] == dim)
        ]
        
        recurrent_group = df_recurrent[
            (df_recurrent['batch'] == batch) & 
            (df_recurrent['length'] == length) & 
            (df_recurrent['heads'] == heads) & 
            (df_recurrent['dim'] == dim)
        ]
        
        # Roofline plot
        intensity_range = np.logspace(0, 2, 100)
        roofline = np.minimum(peak_flops, intensity_range * peak_bandwidth)

        plt.figure(figsize=(8, 6), dpi=150)
        plt.loglog(intensity_range, roofline, linewidth=3)
        
        # Plot arithmetic intensity lines for parallel data
        for ai in parallel_group["arithmetic-intensity"].tolist():
            plt.axvline(ai, color="skyblue", linestyle=':')
        
        # Plot arithmetic intensity lines for recurrent data
        for ai in recurrent_group["arithmetic-intensity"].tolist():
            plt.axvline(ai, color="salmon", linestyle=':')

        plt.xlabel("Arithmetic Intensity (FLOPs/byte)")
        plt.ylabel("Performance (FLOPs/s)")
        plt.title(f"Roofline Model - {group_label}")
        plt.ylim(1e9, peak_flops * 2)
        plt.xlim(1, 1e2)
        plt.tight_layout()
        
        # Save plot to plots directory
        plt.savefig(f'plots/roofline_2_{group_label.lower().replace(" ", "_")}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def make_sram_roofline_plot():
    # Hardware parameters for NVIDIA A100 (FP32)
    peak_flops = 19.5e12  # 19.5 TFLOPS
    peak_bandwidth = 1.555e12  # 1555 GB/s

    # Get unique groups from either dataframe (they should have the same groups)
    groups = df_parallel.groupby(['batch', 'length', 'heads', 'dim'])
    
    # Create one plot per group
    for (batch, length, heads, dim), _ in groups:
        group_label = f'B{batch}L{length}H{heads}D{dim}'
        
        # Get the data for this group from both dataframes
        parallel_group = df_parallel[
            (df_parallel['batch'] == batch) & 
            (df_parallel['length'] == length) & 
            (df_parallel['heads'] == heads) & 
            (df_parallel['dim'] == dim)
        ]
        
        recurrent_group = df_recurrent[
            (df_recurrent['batch'] == batch) & 
            (df_recurrent['length'] == length) & 
            (df_recurrent['heads'] == heads) & 
            (df_recurrent['dim'] == dim)
        ]
        
        # Calculate performance using roofline model
        parallel_performance = np.minimum(
            peak_flops,
            parallel_group['arithmetic-intensity'] * peak_bandwidth
        )
        
        recurrent_performance = np.minimum(
            peak_flops,
            recurrent_group['arithmetic-intensity'] * peak_bandwidth
        )

        plt.figure(figsize=(10, 6))
        
        # Plot the roofline
        sram_range = np.logspace(3, 5, 100)  # SRAM size range from 1KB to 100KB
        plt.axhline(y=peak_flops, color='r', linestyle='--', label='Peak Compute')
        #plt.axhline(y=peak_bandwidth, color='b', linestyle='--', label='Peak Bandwidth')
        
        # Plot parallel model points
        plt.scatter(parallel_group['SRAM-size'], parallel_performance, 
                   color='blue', label='Parallel', marker='o')
        
        # Plot recurrent model points
        plt.scatter(recurrent_group['SRAM-size'], recurrent_performance, 
                   color='red', label='Recurrent', marker='s')

        plt.xlabel('SRAM Size (bytes)')
        plt.ylabel('Performance (FLOPs/s)')
        plt.title(f'SRAM Roofline - {group_label}')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot to plots directory
        plt.savefig(f'plots/sram_roofline_{group_label.lower().replace(" ", "_")}.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        break

# Create plots for the parallel and recurrent models SRAM to AI 
# plot_all_groups(df_parallel, "Parallel")
#plot_all_groups(df_recurrent, "Recurrent")

# Create roofline plots
# make_roofline_plot()

# Create SRAM roofline plots
make_sram_roofline_plot()