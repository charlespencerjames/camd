import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from corr import CorrelationAnalyzer, DataLoader, DataProcessor
from eval import Config


class DataDisplayer:
    """
    Handles displaying data (plots) using the results from correlation DataFrames and dictionaries.
    """
    def __init__(self, processor: DataProcessor, analyzer: CorrelationAnalyzer, config: Config):
        self.processor = processor
        self.analyzer = analyzer
        self.config = config

    def barplot(self, corr_df):
        """
        Display a bar plot for the given correlation DataFrame.
        """
        # Extract number of rows for dynamic plot parameter adjustment
        num_rows = corr_df.shape[0]

        fig_width = min(10, 6 + num_rows / 2.5)  # Dynamically adjust figure width
        fig_height = min(6, 2 + num_rows / 2.5) # Dynamically adjust figure height

        annotation_font_size = max(2, 12 - num_rows / 10) # Dynamically adjust font size

        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.barplot(
            x='Correlation',
            y='Metric',
            data=corr_df,
            color='navy',
        )
        for bar in ax.patches:
            bar.set_height(0.8)
        
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min

        offset_fraction = 0.005
        offset = offset_fraction * x_range

        # Set padding based on absolute positive/negative correlation values
        corr_min = corr_df['Correlation'].min()
        corr_max = corr_df['Correlation'].max()
        padding = 0.04 * (abs(corr_max - corr_min)) 
        padding = offset_fraction ** x_range + padding - padding ** 2
        if corr_min < 0 and corr_max < 0:
            ax.set_xlim(corr_min - padding, 0)
        elif corr_min > 0 and corr_max > 0:
            ax.set_xlim(0, corr_max + padding)
        else:
            ax.set_xlim(corr_min - padding, corr_max + padding)
        
        # Refresh limits in the case they were reset
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min

        # Plot text labels at the end of each bar
        for index, row in corr_df.iterrows():
            row_val = row['Correlation']
            if row_val >= 0:
                x_pos = row_val + offset
                ha = 'left'
            else:
                x_pos = row_val - offset
                ha = 'right'

            ax.text(
                x_pos,
                index,
                f"{row_val:.2f}",
                va='center',
                ha=ha,
                fontsize=annotation_font_size
            )

        if x_min < 0 and x_max > 0:
            plt.axvline(
                0,
                color='purple',
                linestyle='--',
                linewidth=1,
                label="Zero Threshold"
            )
            plt.legend(fontsize=annotation_font_size)

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + y_max * 0.2)

        plt.yticks(fontsize=annotation_font_size)
        plt.xticks(fontsize=annotation_font_size)
        plt.xlabel('Correlation Coefficient', fontsize=(max(8, annotation_font_size))),
        plt.ylabel(''),
        plt.title(
            f'Model: {self.config.model_id}',
            fontsize=max(8, annotation_font_size),
            pad=5
        )

        plt.tight_layout()
        plt.show()

    def roc(self, metric):
        """
        Plot ROC curves for the given DataFrame's metrics against a binary 'Correct' vector.
        """
        df = self.processor.create_dataframe()

        if metric not in df.columns:
            raise ValueError(
                "InputError: Positional argument 'metric' must be a title in DataFrame columns"
            )

        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(df['Correct'], df[metric])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2) # plot chance line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(f'False Positive Rate (Model: {self.config.model_id})')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {metric}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def scores_comparison(
        self, 
        logits=None, 
        softmax=None, 
        norm_logits=None, 
        idx_size: int = 1000, 
        SxL=False, 
        SqLn=False, 
        SdLn=False,
        random_experiment=False
    ):
        """   
        Compares a sample size of `idx_size` of scores between
        logits, softmax, and normalized logits for provided token scores.
        Selecting `True` for `SxL`, `SqLn`, or `SdLn` will display
        the comparisons based on the corresponding mathematical operation.

        :param: `random_experiment` produces random scores for sampling.
        """
        flags = [
            SxL,
            SqLn,
            SdLn
        ]
        if sum(flags) > 1 or sum(flags) == 0:
            raise ValueError(
                "Must provide one from the following: 'SxL', 'SqLn', or 'SdLn', but not multiple."
            )        

        # Random experimentation overrides scores inputs
        if random_experiment:
            vocab_size = 50258 # Arbitrary
            logits = np.random.randn(vocab_size)
            softmax = np.exp(logits) / np.sum(np.exp(logits))
            if not SxL:
                norm_logits = logits / np.sum(logits)
                logits = None

        else:
            kwargs = [
                logits,
                softmax,
                norm_logits
            ]
            if len([1 for scores in kwargs if scores is not None]) != 2:
                raise ValueError(
                "Must provide a pair of keyword arguments, 'logits' and 'softmax' "
                "for SxL, or 'softmax' and 'norm_logits' for SqLn and SdLn.")

            # Variable initialization
            logits = logits
            softmax = softmax
            norm_logits = norm_logits
            vocab_size = len(softmax)

        # Calculate means to adjust for case of negative logits
        logits_mean = np.mean(logits) if logits is not None else None
        softmax_mean = np.mean(softmax)
            
        if SxL:
            if not (logits is not None and softmax is not None):
                raise ValueError(
                    "The softmax logit scores product (SxL) requires 'logits' and 'softmax' inputs."
                )
            combined = logits * softmax
        else:
            if not (softmax is not None and norm_logits is not None):
                raise ValueError(
                    "SqLn and SdLn require 'softmax' and 'normalized logits' inputs."
                )
            elif SqLn:
                combined = softmax / norm_logits
            else:
                combined = softmax - norm_logits

        # Generate sample indices (controlling for convenient display size)
        rng = np.random.default_rng(seed=194) # 194 #83 #954
        sample_indices = rng.choice(vocab_size, size=idx_size, replace=False)  # Randomly sample indices
        sample_indices.sort()

        sample_size = np.arange(idx_size)

        plt.figure(figsize=(12, 8))

        # Logits
        plt.subplot(3, 1, 1)
        plt.plot(
            sample_size, 
            logits[sample_indices] if SxL else norm_logits[sample_indices], 
            label="Logits" if SxL else "Normalized Logits", 
            color='blue')
        if SxL:
            plt.title(f"Logits")
            plt.ylabel("Density")
        else:
            plt.title(f"Normalized Logits")
            plt.ylabel(f"Normalized density")
        plt.grid(True)
 
        # combined of Logits and Softmax Probabilities
        plt.subplot(3, 1, 3)
        plt.plot(sample_size, combined[sample_indices], label="combined", color='green')
        if SxL:
            plt.title("Softmax Logits Product (SxL)")
            plt.ylabel("Probability scaled density")
        elif SqLn:
            plt.title("Softmax Normalized Logits Quotient (SqLn)")
            plt.ylabel("Probabilities to density")
        else:
            plt.title("Softmax Normalized Logits Difference (SdLn)")
            plt.ylabel("Probabilities subtracted by density")
        plt.xlabel(f"Vocabulary Index (Sample Size: {idx_size} / Vocab Size: {vocab_size}) | (Model: {self.config.model_id})")
        plt.grid(True)

        # Retrieve the y-axis limits of subplot 3
        combined_ylim = plt.gca().get_ylim()
        if SxL:
            if softmax_mean < abs(logits_mean * 1e-8):
                ymax = combined_ylim[1] * 5e-2 # Adjust SxL subplot's ymax to 5% of combined ymin for when logits_mean > 0
                ymin = combined_ylim[0] * 5e-2 # Adjust SxL subplot's ymin to 5% of combined ymin for when logits_mean < 0
            elif softmax_mean < abs(logits_mean * 1e-7):
                ymax = combined_ylim[1] * 1e-1 # Adjust SxL subplot's ymax to 10% of combined ymin for when logits_mean > 0
                ymin = combined_ylim[0] * 1e-1 # Adjust SxL subplot's ymin to 10% of combined ymin for when logits_mean < 0
            elif softmax_mean < abs(logits_mean * 1e-6):
                ymax = combined_ylim[1] * 5e-1 # Adjust SxL subplot's ymax to 50% of combined ymin for when logits_mean > 0
                ymin = combined_ylim[0] * 5e-1 # Adjust SxL subplot's ymin to 50% of combined ymin for when logits_mean < 0
            else:
                ymax = combined_ylim[1]  # Adjust SxL subplot's ymax to combined ymin for when logits_mean > 0
                ymin = combined_ylim[0]  # Adjust SxL subplot's ymin to combined ymin for when logits_mean < 0
        elif SqLn:
            ymax = combined_ylim[1] * 5e-4 # Adjust SqLn subplot's ymax to 0.05% of combined ymax for scale
        else:
            ymax = combined_ylim[1] * 5e-2 # Adjust SqLn subplot's ymax to 5% of combined ymax for scale
 
        # Softmax Probabilities
        plt.subplot(3, 1, 2)
        plt.plot(sample_size, softmax[sample_indices], label="Softmax Probabilities", color='orange')
        plt.title("Softmax Probabilities")
        plt.ylabel("Probability")

        # Get softmax_ylim for consistent ymin and ymax depending on logits mean sign
        softmax_ylim = plt.gca().get_ylim() 
        # Set the y-axis limits to compare to subplot 3 depnding on logits mean sign (positive or negative)
        if logits_mean is not None and logits_mean < 0: 
            plt.ylim(ymin, softmax_ylim[1])
        else:
            plt.ylim(softmax_ylim[0], ymax) 
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def metric_comparison(
        self, 
        x_1: str, 
        x_2: str=None, 
        x_3: str=None, 
        idx_size: int = 20, 
        range_2=False, 
        logits=False, 
        softmax=False, 
        negentropy=False,
        SxL=False
    ):

        """   
        Compares metrics element-wise (per token) on the same graph
        with a sampled distribution from the vocabulary size of `idx_size`
        and with binary correct and incorrect values to display
        how the various metrics relate with correctness.
        """
        df = self.processor.create_dataframe()
        binary = df['Correct']
        x_1 = df[x_1]

        if not (len(binary) == len(x_1)):
            raise ValueError("All input arrays must have the same length.")

        labels = sum([
            logits,
            softmax,
            SxL
        ])
        if labels > 1:
            raise ValueError(
                "Cannot input 'True' for more than one flag, 'logits', 'softmax', 'SxL'."
            )

        if x_2:
            x_2 = df[x_2]
        if x_3:
            x_3 = df[x_3]

        def _array_range_adjuster(array):
            array_max = array.max()
            array_min = array.min()
            array_norm = (array - array_min) / (array_max - array_min)
            return (array_norm * 2) - 1 if range_2 else array_norm

        x_1 = _array_range_adjuster(x_1)
        if x_2 is not None:
            x_2 = _array_range_adjuster(x_2)
        if x_3 is not None:
            x_3 = _array_range_adjuster(x_3)
        
        # Randomly sample indices
        rng = np.random.default_rng(seed=77) # seed=84 (else) falcon3 10b # seed=393 (logits) falcon3 10b # seed=3 (SxL) llama 8b # seed=77 llama 8b (SNsr)
        sample_indices = rng.choice(np.arange(len(binary)), size=idx_size, replace=False)  # Randomly sample indices
        sample_indices.sort()

        # Reduce data to sampled indices
        sampled_binary = binary[sample_indices]
        sampled_x_1 = x_1[sample_indices]
        sampled_x_2 = x_2[sample_indices] if x_2 is not None else None
        sampled_x_3 = x_3[sample_indices] if x_3 is not None else None

        # Bar plot for the binary correctness values
        correct_indices = sampled_binary == 1
        incorrect_indices = sampled_binary == 0

        x_positions = np.arange(idx_size)

        # Determine bar heights: +1 for correctness=1, -1 for correctness=0
        bar_heights = np.where(sampled_binary == 1, 1, -1 if range_2 else 0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1) Bar plot for the binary:
        #    Correct (green bars)
        ax.bar(x_positions[correct_indices],
               bar_heights[correct_indices],
               width=0.8,
               color=(0.5, 0.8, 0.5, 1),
               alpha=0.4,
               label='Correct (1)')
        #     Incorrect (red bars)
        ax.bar(x_positions[incorrect_indices],
               bar_heights[incorrect_indices],
               width=0.8,
               color=(1, 0.5, 0.5, 1),
               alpha=0.4,
               label='Incorrect (0)')
        
        # Draw a horizontal line at y=0 for clarity
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

        # 2) Overlay line plots for the correlation values
        if logits:
            color_1 = (0, 0, 0.5, 1)
            color_2 = (0.1, 0.1, 1, 0.5)
            color_3 = (0.65, 0.65, 0.8, 0.6)
            label_1 = 'LMadMn (Logits Max Mean Diff)'
            label_2 = 'LMa (Logits Max)'
            label_3 = 'LMn (Logits Mean)'
        elif softmax:
            color_1 = (0.5, 0.3, 0, 1)     
            color_2 = (1, 0.6, 0, 0.5)    
            color_3 = (0.8, 0.65, 0.45, 0.6) 
            label_1 = 'SVs (Softmax Var Squared)'
            label_2 = 'SV (Softmax Var)'
            label_3 = 'SVsr (Softmax Var Sqrt)'
        elif negentropy:
            color_1 = (0.5, 0, 0.5, 1)  # Dark purple
            color_2 = (0.6, 0.1, 0.9, 0.5)  # Light purple
            color_3 = (0.75, 0.65, 0.8, 0.6)  # Soft lavender
            label_1 = 'SNsr (Softmax Negentropy Sqrt)'
            label_2 = 'SN (Softmax Negentropyx)'
            label_3 = 'SNs (Softmax Negentropy Squared)'
        elif SxL:
            color_1 = (0.85, 0.7, 0.9, 0.8)
            color_2 = 'orange'
            color_3 = 'purple'
            label_1 = 'VSxL'
            label_2 = 'SVs'
            label_3 = 'SxLV'
        else:
            color_1 = 'blue'
            color_2 = 'orange'
            color_3 = 'purple'
            label_1 = 'Logits MAMED'
            label_2 = 'SV'
            label_3 = 'SNsr'
        ax.plot(x_positions, sampled_x_1, marker='o', color=color_1, label=label_1)
        if x_2 is not None:
            ax.plot(x_positions, sampled_x_2, marker='o', color=color_2, label=label_2)
        if x_3 is not None:
            ax.plot(x_positions, sampled_x_3, marker='o', color=color_3, label=label_3)

        # Set y-limits for better visibility
        ax.set_ylim(-1.05 if range_2 else -0.05, 1.05)

        ax.set_xlabel(f'MMLU Question/Answer (Token) Index  (Model: {self.config.model_id})')
        ax.set_ylabel('')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sample_indices)
        ax.tick_params(axis='x', labelsize=8)

        ax.legend()
        plt.title("Scaled Statistical Metric Approximation to Correctness")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Assign class instances
    config = Config()
    loader = DataLoader(config)
    processor = DataProcessor(loader, config)
    analyzer = CorrelationAnalyzer(loader, processor)
    displayer = DataDisplayer(processor, analyzer, config)

    # Load data
    loader.load_data()

    # Retrieve desired dataframes
    pbc_df = analyzer.calculate_correlations()
    top_df, bottom_df = analyzer.get_extremities(pbc_df, extremity_n=5)
    var_df = analyzer.filter_metrics(pbc_df, 'V')
    max_squared_df = analyzer.filter_metrics(pbc_df, 'Ma', ends_with='s')

    #displayer.barplot(pbc_df)
    #displayer.barplot(top_df)
    #displayer.barplot(bottom_df)
    #displayer.barplot(var_df)
    #displayer.barplot(max_squared_df)

    #displayer.roc('SNsr')
    #displayer.roc('SxLVs')

    "------------------------------------------------------------------"
    " Compares metrics element-wise; inputs are their string names     "
    "------------------------------------------------------------------"
    """
    displayer.metric_comparison("LMadMn", x_2="SV", x_3="SNsr", idx_size=20, range_2=True)
    displayer.metric_comparison("LMadMn", x_2="LMa", x_3="LMn", idx_size=20, range_2=True, logits=True)
    displayer.metric_comparison("SVs", x_2="SV", x_3="SStsr", idx_size=20, range_2=True, softmax=True)
    displayer.metric_comparison(x_1='SNsr', x_2='SN', x_3='SNs', idx_size=20, range_2=True, negentropy=True)
    """

    "------------------------------------------------------------------"
    " Score names are converted to load specific scores, 'pos' for     "
    " most correlated, 'neg' for most negatively correlated, and       "
    " 'zero' for least correlated. The data is saved in corr.py        "
    "------------------------------------------------------------------"
    max_scores = loader.load_scores(filename=f'{config.model_path.replace('stats.h5','pos.npz')}')
    zero_scores = loader.load_scores(filename=f'{config.model_path.replace('stats.h5','zero.npz')}')
    min_scores = loader.load_scores(filename=f'{config.model_path.replace('stats.h5','neg.npz')}')
    
    "------------------------------------------------------------------"
    " Compares scores for provided token data of and selected operation"
    " `random_experiment` produces random scores for sampling.          "
    "------------------------------------------------------------------"
    #displayer.scores_comparison(logits=min_scores['logits'], softmax=min_scores['softmax'], idx_size=1000, SxL=True, random_experiment=False)




    

    




