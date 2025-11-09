import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_comparison(topk_csv, random_csv, output_dir="."):
    """
    Reads and plots a comparison between top-k and random prototype masking.

    Args:
        topk_csv (str): Path to the mask_topk_summary.csv file.
        random_csv (str): Path to the random_mask_summary.csv file.
        output_dir (str): Directory to save the plot.
    """
    try:
        df_topk = pd.read_csv(topk_csv)
        df_random = pd.read_csv(random_csv)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Validate columns
    if 'mask_topk' not in df_topk.columns or 'accuracy_top1' not in df_topk.columns:
        print(f"Error: {topk_csv} must contain 'mask_topk' and 'accuracy_top1' columns.")
        return
    if 'random_mask_k' not in df_random.columns or 'accuracy_top1' not in df_random.columns:
        print(f"Error: {random_csv} must contain 'random_mask_k' and 'accuracy_top1' columns.")
        return

    plt.figure(figsize=(10, 7))

    # Plot Top-K masking results
    plt.plot(df_topk['mask_topk'], df_topk['accuracy_top1'], marker='o', linestyle='-', label='Top-K Masking')
    # Plot Random masking results
    plt.plot(df_random['random_mask_k'], df_random['accuracy_top1'], marker='x', linestyle='--', label='Random Masking')

    plt.title('Top-K vs. Random Prototype Masking Impact on Accuracy')
    plt.xlabel('Number of Masked Prototypes (k)')
    plt.ylabel('Top-1 Accuracy')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    all_k_values = sorted(list(set(df_topk['mask_topk']) | set(df_random['random_mask_k'])))
    plt.xticks(all_k_values)
    
    output_filename = os.path.join(output_dir, 'masking_comparison_plot.png')
    plt.savefig(output_filename)
    print(f"Comparative plot saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a comparative plot for prototype masking experiments."
    )
    parser.add_argument(
        '--topk_csv', 
        type=str, 
        default='mask_topk_summary.csv',
        help='Path to the summary CSV from top-k masking.'
    )
    parser.add_argument(
        '--random_csv', 
        type=str, 
        default='random_mask_summary.csv',
        help='Path to the summary CSV from random masking.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='.',
        help='Directory to save the generated plot.'
    )
    args = parser.parse_args()

    plot_comparison(args.topk_csv, args.random_csv, args.output_dir)