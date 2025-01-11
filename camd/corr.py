import os
from typing import Union
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from config import Config

class DataLoader:
    """
    Loads and prepares data from the given HDF5 file.
    """
    def __init__(self, config: Config):
        self.config = config
        self.main_path = config.main_path
        self.second_path = config.second_path
        self.dir_path = config.dir_path

        self.num_epochs = config.num_epochs
        self.seq_per_epoch = config.seq_per_epoch

        self.match_or_mismatch_total = None # the total of correct and incorrect answers
        self.tok_data = None
        self.tok_scores = None
        self.tok_npz = None

    def load_data(self):
        """ 
        Load `match_or_mismatch_total` and `tok_data` from an HD5F file.
        Loads `tok_scores` separately if `max_new_tokens == 1`
        """
        with h5py.File(self.main_path, "r") as f:
            self.match_or_mismatch_total = [
                s.decode('utf-8') for s in f["match_or_mismatch_total"][:]
            ]
            self.tok_data = {
                key: f[key][:] for key in f.keys() if key != "match_or_mismatch_total"
            }
        if self.config.max_new_tokens == 1:
            with h5py.File(self.second_path, "r") as f:
                self.tok_scores = {key: f[key][:] for key in f.keys()}
    
    def save_scores(
        self, 
        index: int, 
        scores: dict = None, 
        filename='scores.npz'
    ):
        """ 
        Saves designated scores as a numpy `.npz` file.
        """
        if scores is None:
            scores = self.tok_scores

        def _organize_scores(scores, index):
            return {key: value[index] for key, value in scores.items()}

        organized_scores = _organize_scores(scores, index)
        np.savez(os.path.join(self.dir_path, filename), **organized_scores)


    def load_scores(self, filename='scores.npz') -> dict:
        """
        Load scores from a numpy `.npz` file and return them as a dictionary.
        """
        filepath = os.path.join(self.dir_path, filename)
        npz = np.load(filepath)
        return {key: npz[key] for key in npz}

    def retrieve_scores(
        self, 
        index: int, 
        score_type: str = None, 
    ) -> dict:
        """ 
        Retrieves the scores from designated `tok_scores` as `score_type` of 
        (logits, softmax probabilities, or normalized logits). If no `score_type`
        is provided, returns all types as a dictionary.
        """
        # Validate the index
        max_size = len(next(iter(self.tok_scores.values())))
        if index > max_size:
            raise IndexError(
                f"'Index {index}' is out of range. Maximum size is {max_size}." 
                )

        # Validate the score type
        if score_type is not None:
            score_type = score_type.lower().strip()
            valid_types = {'logits', 'softmax', 'normalized logits'}
            if score_type not in valid_types:
                raise ValueError(
                    f"'{score_type}' is not a valid score type. Choose from: {valid_types}."
                   )
            return self.tok_scores[score_type][index]

        # Return all scores for the index as a dictionary
        return {key: value[index] for key, value in self.tok_scores.items()}
        
    def get_answers(
        self, 
        bool=False, 
        array=False, 
        correct_ratio=False,
        answered_ratio=False
    ) -> Union[list, np.array, float]:
        """
        Convert `match_or_mismatch_total` to a boolean list, numpy array, 
        ratio of correct answers, or ratio of answered questions.
        """
        # Validate input flags
        labels = [bool, array, correct_ratio, answered_ratio]
        if sum(labels) > 1 or sum(labels) == 0:
            raise ValueError(
                    "Must choose no more than one kwarg from 'bool', "
                    "'array', 'correct_ratio', or 'answered_ratio'."
            )

        # Convert strings to a boolean list ('s' for 'string')
        match_or_mismatch_bool = [
            s == 'True' for s in self.match_or_mismatch_total if s in ['True', 'False', 'NA']
            ]

        if len(match_or_mismatch_bool) != len(self.match_or_mismatch_total):
            raise ValueError(
                "There is an error with collecting answers for match_or_mismatch_total. "
                "Refer to answer collection in the 'run_inference' method in the "
                "'ModelInference' class in 'eval.py' to address the problem."
                )

        # Handle the requested output type
        if bool:
            return match_or_mismatch_bool
        elif array:
            return np.array(match_or_mismatch_bool).astype(int)
        elif correct_ratio:
            return sum(match_or_mismatch_bool) / len(match_or_mismatch_bool)
        elif answered_ratio:
            match_or_mismatch_NA = [
                s == 'NA' for s in self.match_or_mismatch_total
            ]
            NA_ratio = sum(match_or_mismatch_NA) / len(match_or_mismatch_bool)
            return 1 - NA_ratio
    
    def print_summary(self):
        """Prints a summary of correct/incorrect answers and ratio."""
        match_or_mismatch_bool = self.get_answers(bool=True)
        total_count = len(match_or_mismatch_bool)
        ans_ratio = self.get_answers(answered_ratio=True)
        correct_count = sum(match_or_mismatch_bool)
        incorrect_count = total_count - correct_count
        ratio_correct = correct_count / total_count if total_count > 0 else 0.0

        print(f"Ratio of answered questions:......{ans_ratio*1e2:.1f}%")
        print(f"Number of correct answers:........{correct_count}")
        print(f"Number of incorrect answers:......{incorrect_count}")
        print(f"Overal ratio of correct answers:..{ratio_correct*1e2:.1f}%")

class DataProcessor:
    """
    Handles the creation and formatting of the pandas Dataframe from loaded data.
    """
    def __init__(self, loader: DataLoader, config: Config):
        self.loader = loader
        self.answers = None

    def create_dataframe(self) -> pd.DataFrame:
        """Creates and formats a DataFrame with metric data and correctness."""
        temp_tok_data = {
            stat_name: stat for stat_name, stat in self.loader.tok_data.items()
        }
        temp_tok_data['Correct'] = self.loader.get_answers(array=True)

        df = pd.DataFrame(temp_tok_data)
        return df
    
class CorrelationAnalyzer:
    """
    Performs correlation analysis and provides utilities to filter metrics token-wise and element-wise.
    """
    def __init__(self, loader: DataLoader, processor: DataProcessor):
        self.loader = loader
        self.processor = processor

    def calculate_correlations(self, df: pd.DataFrame = None):
        """
        Calculate Point-Biserial correlations of metrics with the 'Correct' column.
        """
        if df is None or df.empty:
            df = self.processor.create_dataframe()
        if 'Correct' not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'Correct' column."
            )
        
        corr_df = {}
        p_values = {}
        for col in df.columns:
            if col == 'Correct':
                continue
            corr, p_val = pointbiserialr(df[col], df['Correct'])
            corr_df[col] = corr
            p_values[col] = p_val
        
        corr_df = pd.DataFrame.from_dict(corr_df, orient='index', columns=['Correlation'])
        corr_df['P-Value'] = corr_df.index.map(p_values)
        corr_df = corr_df.sort_values(by='Correlation', ascending=False).reset_index()
        corr_df.rename(columns={'index': 'Metric'}, inplace=True)

        return corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

    def get_extremities(self, corr_df: pd.DataFrame, extremity_n: int = 10):
        """Returns the top and bottom 'extremity_n' metrics by correlation."""
        if 'Correlation' not in corr_df.columns:
            raise ValueError(
                "Correlation DataFrame must contain a 'Correlation' column."
            )

        top_corr = corr_df.nlargest(extremity_n, 'Correlation')[['Metric', 'Correlation', 'P-Value']]
        bottom_corr = corr_df.nsmallest(extremity_n, 'Correlation')[['Metric', 'Correlation', 'P-Value']].reset_index(drop=True)

        return top_corr, bottom_corr

    def filter_metrics(
        self, 
        corr_df: pd.DataFrame, 
        metric: str = None, 
        starts_with: str = None, 
        ends_with: str = None
    ) -> pd.DataFrame:
        """
        Filter the correlation DataFrame by any and all inputs.
        `metric` checks if the input exists in any 'Metric' column value.
        `starts_with` filters for columns that start with the specified string.
        `ends_with` filters for columns that end with the specified string.
        """
        # Ensure at least one filter criterion is provided
        if not metric and not starts_with and not ends_with:
            raise ValueError(
                "Must provide at least one of the following: 'metric', 'starts_with', 'ends_with'."
            )

        # Helper function to check if all words in `input_str` exist in `column`, ignoring order
        def _in_column(column, input_str):
            column_set = set(column)
            input_set = set(input_str)
            return input_set.issubset(column_set)

        # Apply filters
        if metric:
            corr_df = corr_df[corr_df['Metric'].apply(lambda col: _in_column(col, metric))]
        if starts_with:
            corr_df = corr_df[corr_df['Metric'].str.startswith(starts_with, na=False)]
        if ends_with:
            corr_df = corr_df[corr_df['Metric'].str.endswith(ends_with, na=False)]


        if corr_df.empty:
            raise ValueError("No matching data found based on the provided filters.")

        # Reset index for the resulting DataFrame
        return corr_df.reset_index(drop=True)

    def filter_elements(
        self,
        metric: str = None,
        return_pos_corr=False,
        return_pos_metric=False,
        return_pos_idx=False,
        return_neg_corr=False,
        return_neg_metric=False,
        return_neg_idx=False,
        return_zero_corr=False,
        return_zero_metric=False,
        return_zero_idx=False,
    ) -> Union[float, int, pd.DataFrame]:
        """
        Filters and returns specific elements from the data based on the flags set.
        Only one of the flags can be set to True.
        """
        # Ensure only one flag is True
        flags = [
            return_pos_corr,
            return_pos_metric,
            return_pos_idx,
            return_neg_corr,
            return_neg_metric,
            return_neg_idx,
            return_zero_corr,
            return_zero_metric,
            return_zero_idx,
        ]
        if sum(flags) > 1:
            raise ValueError("Must only input one of the keyword arguments, not multiple.")

        # Create/obtain the DataFrame
        df = self.processor.create_dataframe()

        # Helper function
        def _find_zero_corr(series):
            """Helper to find the value and index in `series` closest to zero."""
            # idxmin on the absolute value returns the integer index in `series`
            idx_nearest = series.abs().idxmin()
            val_nearest = series.loc[idx_nearest]
            return val_nearest, idx_nearest

        #
        # CASE 1: If a specific metric is provided
        #
        if metric:
            # Split by 'Correct' == 1 or 0
            group1 = df.loc[df['Correct'] == 1, metric]
            group0 = df.loc[df['Correct'] == 0, metric]

            # 1. `zero_corr`` finds values closest to zero for all cases 
            #    indicating zero correlation approximation
            g1_zero_val, g1_zero_idx = _find_zero_corr(group1)
            g0_zero_val, g0_zero_idx = _find_zero_corr(group0)

            if abs(g1_zero_val) < abs(g0_zero_val):
                chosen_zero_val, chosen_zero_idx = g1_zero_val, g1_zero_idx
            else:
                chosen_zero_val, chosen_zero_idx = g0_zero_val, g0_zero_idx

            if return_zero_corr:
                return chosen_zero_val
            if return_zero_idx:
                return chosen_zero_idx

            # 2. `pos_corr` finds the value that's approximated to be the most posively correlated
            #    Since correlations depend on the binary state, '0' or '1', both are compared.
            group1_max_val, group1_max_idx = group1.max(), group1.idxmax()
            group0_min_val, group0_min_idx = group0.min(), group0.idxmin()

            if abs(group1_max_val) > abs(group0_min_val):
                chosen_pos_val, chosen_pos_idx = group1_max_val, group1_max_idx
            else:
                chosen_pos_val, chosen_pos_idx = group0_min_val, group0_min_idx

            if return_pos_corr:
                return chosen_pos_val
            if return_pos_idx:
                return chosen_pos_idx

            # 3. `neg_corr` finds the value that's approximated to be the most negatively correlated
            group1_min_val, group1_min_idx = group1.min(), group1.idxmin()
            group0_max_val, group0_max_idx = group0.max(), group0.idxmax()

            if abs(group1_min_val) > abs(group0_max_val):
                chosen_neg_val, chosen_neg_idx = group1_min_val, group1_min_idx
            else:
                chosen_neg_val, chosen_neg_idx = group0_max_val, group0_max_idx

            if return_neg_corr:
                return chosen_neg_val
            if return_neg_idx:
                return chosen_neg_idx

            # 4. By default, if none of the flags match, just return the sorted df
            return df.sort_values(by=metric, ascending=False)

        #
        # CASE 2: If metric is not specified - consider all metric columns
        #
        else:
            metric_cols = [col for col in df.columns if col != 'Correct']

            # Split into group1_df, group0_df
            group1_df = df.loc[df['Correct'] == 1, metric_cols]
            group0_df = df.loc[df['Correct'] == 0, metric_cols]

            # For each column, find the value closest to zero
            g1_zero_vals = group1_df.apply(lambda col: _find_zero_corr(col)[0], axis=0)
            g0_zero_vals = group0_df.apply(lambda col: _find_zero_corr(col)[0], axis=0)

            # Compares the absolute min across group1_zero_vals vs group0_zero_vals
            g1_min_abs_value = g1_zero_vals.abs().min()
            g0_min_abs_value = g0_zero_vals.abs().min()

            if g1_min_abs_value < g0_min_abs_value:
                # Column name with the smallest absolute value in group1
                col_zero = g1_zero_vals.abs().idxmin()
                val_zero = g1_zero_vals[col_zero]
                # Finds index in the full df where that metric is closest to zero
                _, idx_zero = _find_zero_corr(df.loc[df['Correct'] == 1, col_zero])
            else:
                col_zero = g0_zero_vals.abs().idxmin()
                val_zero = g0_zero_vals[col_zero]
                _, idx_zero = _find_zero_corr(df.loc[df['Correct'] == 0, col_zero])

            if return_zero_corr:
                return val_zero
            if return_zero_metric:
                return col_zero
            if return_zero_idx:
                return idx_zero

            # `pos_corr` logic across all columns
            g1_max_val = group1_df.max().max()  # the largest value in group1 across all metrics
            g0_min_val = group0_df.min().min()  # the smallest value in group0 across all metrics

            if abs(g1_max_val) > abs(g0_min_val):
                # find which column in group1_df has the max val
                col_pos_corr = group1_df.max().idxmax()  # gets col with max across group1
                val_pos_corr = group1_df[col_pos_corr].max()
                idx_pos_corr = df[col_pos_corr].idxmax()
            else:
                col_pos_corr = group0_df.min().idxmin()  # gets col with min across group0
                val_pos_corr = group0_df[col_pos_corr].min()
                idx_pos_corr = df[col_pos_corr].idxmin()

            if return_pos_corr:
                return val_pos_corr
            if return_pos_metric:
                return col_pos_corr
            if return_pos_idx:
                return idx_pos_corr

            # `neg_corr`` logic
            g1_min_val = group1_df.min().min()
            g0_max_val = group0_df.max().max()

            if abs(g1_min_val) > abs(g0_max_val):
                col_neg_corr = group1_df.min().idxmin()
                val_neg_corr = group1_df[col_neg_corr].min()
                idx_neg_corr = df[col_neg_corr].idxmin()
            else:
                col_neg_corr = group0_df.max().idxmax()
                val_neg_corr = group0_df[col_neg_corr].max()
                idx_neg_corr = df[col_neg_corr].idxmax()

            if return_neg_corr:
                return val_neg_corr
            if return_neg_metric:
                return col_neg_corr
            if return_neg_idx:
                return idx_neg_corr

            el_df = pd.DataFrame(
                {
                    m: df.sort_values(by=m, ascending=False)[m] 
                    for m in metric_cols
                }
            )
            return el_df

    def print_summary(self):
        pbc_df = self.calculate_correlations()
        pbc_top_corr, pbc_bottom_corr = self.get_extremities(pbc_df, extremity_n=5)
        print(f"Correlation Coefficient Top Metrics:\n{pbc_top_corr}")
        print(f"Correlation Coefficient Bottom Metrics:\n{pbc_bottom_corr}")

if __name__ == "__main__":
    # Assign class instances
    config = Config()
    loader = DataLoader(config)
    processor = DataProcessor(loader, config)
    analyzer = CorrelationAnalyzer(loader, processor)

    # Load Data
    loader.load_data()

    # Print summaries of answer (loader) and correlation (analyzer) data
    loader.print_summary()
    analyzer.print_summary()

    "------------------------------------------------------------------"
    " Calculate and print DataFrame corrleations and extremities       "
    "------------------------------------------------------------------"
    """
    pbc_df = analyzer.calculate_correlations()
    top_pbc, bottom_pbc = analyzer.get_extremities(pbc_df, extremity_n=10) 
    #print(pbc_df)
    #print(top_pbc)
    #print(bottom_pbc)
    """

    "------------------------------------------------------------------"
    " Filter correlations freely based on metric names                 "
    " (Using 'LnV', 'sr', 'SN', and 'S' are purely arbitrary examples) "
    "------------------------------------------------------------------"
    """
    sample_1 = analyzer.filter_metrics(pbc_df, 'LnV', ends_with='sr')
    sample_2 = analyzer.filter_metrics(pbc_df, 'SN', starts_with='S')
    print(sample_1)
    print(sample_2)
    """

    "------------------------------------------------------------------"
    " Extract highest, lowest, and least correlated                    "
    " element-wise approximations from all metrics in token DataFrame  "
    "------------------------------------------------------------------"
    """    
    pos_corr = analyzer.filter_elements(return_pos_corr=True)
    pos_metric = analyzer.filter_elements(return_pos_metric=True)
    pos_idx = analyzer.filter_elements(return_pos_idx=True)
    neg_corr = analyzer.filter_elements(return_neg_corr=True)
    neg_metric = analyzer.filter_elements(return_neg_metric=True)
    neg_idx = analyzer.filter_elements(return_neg_idx=True)
    zero_corr = analyzer.filter_elements(return_zero_corr=True)
    zero_metric = analyzer.filter_elements(return_zero_metric=True)
    zero_idx = analyzer.filter_elements(return_zero_idx=True)
    print(pos_corr)
    print(pos_metric)
    print(pos_idx)
    print(neg_corr)
    print(neg_metric)
    print(neg_idx)
    print(zero_corr)
    print(zero_metric)
    print(zero_idx)
    """

    "------------------------------------------------------------------"
    " Extract highest, lowest, and least correlated                    "
    " element-wise approximations based on input `metric`.             "
    " (Using 'LdSxLMn' as an arbitrary example)                        "
    "------------------------------------------------------------------"
    
    metric_pos_corr = analyzer.filter_elements(metric='SNsr', return_pos_corr=True)
    metric_pos_idx = analyzer.filter_elements(metric='SNsr', return_pos_idx=True)
    metric_neg_corr = analyzer.filter_elements(metric='SNsr', return_neg_corr=True)
    metric_neg_idx = analyzer.filter_elements(metric='SNsr', return_neg_idx=True)
    metric_zero_corr = analyzer.filter_elements(metric='SNsr', return_zero_corr=True)
    metric_zero_idx = analyzer.filter_elements(metric='SNsr', return_zero_idx=True)
    print(metric_pos_corr)
    print(metric_pos_idx)
    print(metric_neg_corr)
    print(metric_neg_idx)
    print(metric_zero_corr)
    print(metric_zero_idx)
    
    
    "------------------------------------------------------------------"
    " Save most positively, negatively, and least correlated           "
    " element-wise approximations to current working directory, and    "
    " then load and test print their values.                           "
    "------------------------------------------------------------------"
    """
    loader.save_scores(index=metric_pos_idx, filename='pos.npz')
    loader.save_scores(index=metric_neg_idx, filename='neg.npz')
    loader.save_scores(index=metric_zero_idx, filename='zero.npz')
    pos_npz = loader.load_scores(filename='pos.npz')
    neg_npz = loader.load_scores(filename='neg.npz')
    zero_npz = loader.load_scores(filename='zero.npz')
    print(pos_npz)
    print(neg_npz)
    print(zero_npz)
    """

