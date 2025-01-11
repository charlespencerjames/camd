import os
import time
import gc
import re
import textwrap
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config
from tensors import initialize_tensors
import mcsi

###############################################
# Data Preparation
###############################################

class DataPreparer:
    def __init__(self, config: Config):
        self.config = config

    def load_datasets(self, folder_path):
        """ Load all CSV files in the folder and return a list of DataFrames."""
        ds_file_pairs = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ds = pd.read_csv(file_path, header=None)
            ds_file_pairs.append((filename, ds))
        return ds_file_pairs

###############################################
# Prompt Generation
###############################################

class PromptGenerator:
    def __init__(self, data_preparer: DataPreparer, config: Config):
        self.config = config
        self.data_preparer = data_preparer
        self.folder_path_test = self.config.folder_path_test
        self.folder_path_val = self.config.folder_path_val

        # Load datasets
        self.ds_file_pairs_test = sorted(self.data_preparer.load_datasets(self.folder_path_test), key=lambda x: x[0])
        self.ds_file_pairs_val = sorted(self.data_preparer.load_datasets(self.folder_path_val), key=lambda x: x[0])

        # Align datasets by their file names
        self.ds_dict_test = {}
        self.ds_dict_val = {}

        for test_file, test_ds in self.ds_file_pairs_test:
            val_file, val_ds = next(
                ((v_file, v_ds) for v_file, v_ds in self.ds_file_pairs_val if v_file.replace('_val', '_test') == test_file),
                (None, None)
            )
            if val_ds is None:
                raise ValueError(f"Missing matching validation dataset for test file: {test_file}")
            
            self.ds_dict_test[test_file] = test_ds
            self.ds_dict_val[val_file] = val_ds

        self.instruct = self.config.instruct
        self.tool_call = self.config.tool_call
        self.full_mmlu_override = self.config.full_mmlu_override

    def generate_shot_prompts(self, epoch_idx):
        """
        Generate shot prompts for few-shot inference.
        Each dataset in folder_path is used to create prompt sequences.
        """
        all_shots = []
        for filename_val, ds_val in self.ds_dict_val.items():
            # Find corresponding test dataset by filename
            filename_test = filename_val.replace('_val', '_test')
            ds_test = self.ds_dict_test.get(filename_test)

            if ds_test is None:
                raise ValueError(f"No matching test dataset found for validation file: {filename_val}")

            # Extract subject from filename to handle special cases for subject naming
            subject = filename_val.replace('_val.csv', '').replace('_', ' ')
            if subject in ['management', 'miscellaneous']:
                subject += ' topics'

            # Prepare the prefix prompt
            if self.instruct:
                prefix = {"role": "system", "content": f"You are a test-taker, so choose the correct answer to this question about {subject}"}
            else:
                prefix = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            if self.tool_call:
                prefix = ['<tool_call>[]</tool_call>', prefix]

            # Convert dataframe to list of rows
            rows_test = list(ds_test.iterrows())
            rows_val = list(ds_val.iterrows())
            shot_prompts = []
            seq_range = range(len(rows_test)) if self.full_mmlu_override else range(self.config.seq_per_dataset)

            for seq_idx in seq_range:
                shots = [] # The epoch_idx expression prevents repetition of consecutive shorts over epochs while the seq_idx expression iterates over each shot sequentially
                start_idx = ((epoch_idx * self.config.seq_per_epoch * self.config.num_shots) + (seq_idx * self.config.num_shots)) % len(rows_val) # Allows for unlimited dataframe loops
                
                for i in range(self.config.num_shots):
                    idx = (start_idx + i) % len(rows_val)
                    row = rows_val[idx][1] # Skips index element in tuple (index, content)
                    q, A, B, C, D, ans = (str(item).strip() for item in row.iloc[:6])
                    
                    if self.instruct:
                        shots.append({"role": "user", "content": f"{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"})
                        shots.append({"role": "assistant", "content": f"{ans}\n\n"})
                    else:
                        shots.append(f"{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {ans}\n\n")

                # Insert the prefix at the start of the sequence of shots
                shots.insert(0, prefix)
                shot_prompts.append(shots)
            all_shots.append(shot_prompts)
        return all_shots # Shape: (num_datasets, per-dataset sequences, prefix + num_shots, num_shot strings)

    def generate_test_prompts(self, epoch_idx):
        """
        Generate test prompts for evaluation.
        Returns 'test_prompts' and 'all_answers'.
        """
        test_prompts = []
        all_answers = []
        for _, ds_test in self.ds_dict_test.items():
            rows = list(ds_test.iterrows())
            prompts = []
            answers = []

            if self.full_mmlu_override:
                if epoch_idx * self.config.seq_per_epoch > len(rows):
                    raise IndexError(f"Requested sequence iterations per epoch is out of bounds: {epoch_idx} * {self.config.seq_per_epoch}")

            start = (epoch_idx * self.config.seq_per_epoch)
            seq_range = len(rows) if self.full_mmlu_override else self.config.seq_per_dataset
            i = 0
            while i < seq_range:
                row = rows[(start + i) % len(rows)][1]
                q, A, B, C, D, ans = (str(item).strip() for item in row.iloc[:6])
                if self.instruct:
                    prompts.append({"role": "user", "content": f"{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"})
                else:
                    prompts.append(f"{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: ")
                answers.append(ans)
                i += 1
            test_prompts.append(prompts)
            all_answers.append(answers) # Each element is a single test prompt string
            flattened_answers = [ans for seq in all_answers for ans in seq]
        return test_prompts, flattened_answers # Shape: (num_datasets, per-dataset sequences, test_prompts string/single character answer)

    def concatenate(self, epoch_idx):
        """
        Combine shot prompts and test prompts into full prompts.
        """
        shot_prompts = self.generate_shot_prompts(epoch_idx)
        test_prompts, answers = self.generate_test_prompts(epoch_idx)
        if len(shot_prompts) != len(test_prompts):
            raise ValueError(f"Mismatch in lengths of shot_prompts ({len(shot_prompts)}) and test_prompts ({len(test_prompts)}).")

        for dataset_idx, dataset in enumerate(test_prompts):
            for seq_idx, seq in enumerate(dataset):
                # Append test prompt to the corresponding shot prompt
                shot_prompts[dataset_idx][seq_idx].append(seq) 
        if not self.instruct:
            shot_prompts = [[''.join(seq) for seq in dataset] for dataset in shot_prompts] # Shape: (num_datasets, per-dataset sequences, full prompt strings)
        prompts = [seq for dataset in shot_prompts for seq in dataset] # Shape: ((total number of flattened) prompts, full prompt strings / dictionaries (if 'self.instruct'))
        return prompts, answers

###############################################
# Model Inference and Metrics Collection
###############################################

class ModelInference:
    def __init__(self, config: Config):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, device_map='auto').to(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, padding_side='left', clean_up_tokenization_spaces=True)
        if self.model.config.pad_token_id:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        if self.config.instruct:
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = """
                    {%- for message in messages -%}
                        {{- '<|' + message['role'] + '|>' -}}
                        {{- message['content'] + eos_token -}}
                    {%- endfor -%}
                    {%- if add_generation_prompt -%}
                        {{- '<|assistant|>' -}}
                    {%- endif -%}
                    """ # Add jinja template if None

        self.instruct = self.config.instruct
        self.full_mmlu_override = self.config.full_mmlu_override
        self.model.eval()

    def collate_fn(self, prompts):
        """
        Collate_fn is called by dataloader for every sequence in the for loop below.
        This means each output will only be for one sequence and thus the shape illustrated below.
        """
        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        return prompt_tokens # Shape (input_ids + attention_mask (== 2), tokens (instead of strings))
    
    def run_inference(self, prompts, epoch_idx, answers, match_or_mismatch_total, stats_collector=None, total_seq_batches=None):
        """
        Main method in charge of generating tokens, responses, and score tensors for calculations.
        Runs per epoch as designated in configuration and activated in the inference function.
        """
        if self.instruct:
            prompts = self.tokenizer.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        
        # The DataLoader loads only one sequence batch at a time for memory optimization,
        # so it is called to load the next sequence batch by the for loop below for each iteration.
        dataloader = DataLoader(
            prompts,
            collate_fn=self.collate_fn,
            batch_size=self.config.batch_size
        ) # Shape: ((prompts + batch_size - 1) // batch_size, input_ids + attention_mask, batch_size, tokens)  

        response_number = epoch_idx * self.config.seq_per_epoch
        with torch.no_grad(): 
            for seq_idx, seq in enumerate(dataloader): # Shape: (input_ids + attention mask, batch_size, tokens)
                input_ids = seq['input_ids'].to(self.config.device) # Shape: (batch_size, tokens)
                attention_mask = seq['attention_mask'].to(self.config.device) # Shape: (batch_size, tokens)
                prompt_length = input_ids.size(1) # (== token_length (number of characters))

                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    output_scores=True, # Outputs logits of vocab_size per token
                    return_dict_in_generate=True, # Returns dictionary for output scores
                )

                # Shape: (generated_tokens (-> within range of max_new_tokens), batch_size, vocab_size)
                scores = torch.stack(output.scores, dim=0)
                scores = scores.permute(1, 0, 2) # Permute batch_size to first dimension -> shape: (batch_size, generated_tokens, vocab_size)

                current_batch_size, generated_tokens, vocab_size = scores.size() # Get sizes for tensor initiation

                # Handles edge case of batch_size mismatch with last iteration (aligns tensor shapes)
                if current_batch_size < self.config.batch_size:
                    nan_fill = torch.full((self.config.batch_size - scores.size(0), *scores.shape[1:]), float('nan'), device=self.config.device)
                    scores = torch.cat([scores, nan_fill], dim=0)

                # Initiate empty tensors
                logits_tok = torch.full((self.config.batch_size, generated_tokens, vocab_size), float('nan'), device=self.config.device).to(self.config.precision)
                softmax_tok = torch.full((self.config.batch_size, generated_tokens, vocab_size), float('nan'), device=self.config.device).to(self.config.precision)
                norm_logits_tok = torch.full((self.config.batch_size, generated_tokens, vocab_size), float('nan'), device=self.config.device).to(self.config.precision)

                # Assign values
                logits_tok[:] = scores
                softmax_tok[:] = torch.softmax(scores, dim=-1)
                norm_logits_tok[:] = logits_tok / (torch.sum(logits_tok, dim=-1, keepdim=True) + 1e-9)

                # Current index equals current number of completed sequences
                for batch_idx in range(self.config.batch_size):
                    if self.full_mmlu_override:
                        current_seq_batch = epoch_idx * total_seq_batches + seq_idx
                    else:
                        current_seq_batch = epoch_idx * self.config.seq_per_epoch + seq_idx
                    gen_tokens = len(~torch.isnan(logits_tok[batch_idx])) # Track number of generated tokens

                    # Update stats if necessary
                    if stats_collector is not None:
                        stats_collector.update_stats(
                            batch_idx,
                            current_seq_batch,
                            gen_tokens,
                            logits_tok,
                            softmax_tok,
                            norm_logits_tok,
                            vocab_size
                        )
                    
                    # Handles edge case of batch_size mismatch with last iteration (prevents indexing errors)
                    if not self.full_mmlu_override:
                        if self.config.total_seq <= response_number:
                            continue

                    # Decode response
                    response_number += 1
                    response = textwrap.fill(
                        self.tokenizer.decode(output.sequences[batch_idx, prompt_length:], skip_special_tokens=True).strip(),
                        width=50
                    )
                    if self.config.print_responses:
                        print(f"Response {response_number}:\n{response}\n")

                    # Check answer correctness
                    if config.full_mmlu_override:
                        correct_answer = answers[response_number - 1]
                    else:
                        correct_answer = answers[(response_number - 1) % self.config.seq_per_epoch]
                    match_or_mismatch = AnswerChecker.check_answers(response, correct_answer)
                    match_or_mismatch_total.append(match_or_mismatch)

            torch.cuda.empty_cache()
            gc.collect()

###############################################
# Answer Checking
###############################################

class AnswerChecker:
    @staticmethod
    def check_answers(output, answer):
        """
        Check if the model's output contains a single choice [A-D].
        If multiple or none, return 'NA'.
        If correct, return True, else False.
        """
        if re.search(r'\b[A-D]\b', output, re.IGNORECASE):
            pattern = r'\b[A-D]\b'
            regex = re.compile(pattern, re.IGNORECASE)
            match_grp_lst = regex.findall(output)
            if len(match_grp_lst) == 1:
                match_x = match_grp_lst[0].upper()
            else:
                # More than one answer found
                return 'NA'
        else:
            # No matches found
            return 'NA'
        
        return True if match_x == answer else False

###############################################
# Statistics Collector
###############################################

class StatsCollector:
    def __init__(self, config: Config, total_seq_batches=None):
        # Initializes all the tensors just as originally done
        self.config = config
        if config.full_mmlu_override:
            initialize_tensors(config.batch_size, config.max_new_tokens, config.device, config.precision, total_seq_batches=total_seq_batches)
        else:
            initialize_tensors(config.batch_size, config.max_new_tokens, config.device, config.precision, total_seq_batches=config.total_seq_batches)

        """
        Explanations of 'Naming Conventions' and 'Ordering Logic' for the 
        naming system of the variables below can be found at the top of 'tensors.py'
        """

        # Import all references after initialization
        from tensors import (
            # 3 * 7 = 21
            LMn_tok,
            LMe_tok,
            LSt_tok,
            LV_tok,
            LMi_tok,
            LMa_tok,
            LMc_tok,
            SMn_tok,
            SMe_tok,
            SSt_tok,
            SV_tok,
            SMi_tok,
            SMa_tok,
            SMc_tok,
            LnMn_tok,
            LnMe_tok,
            LnSt_tok,
            LnV_tok,
            LnMi_tok,
            LnMa_tok,
            LnMc_tok,

            SE_tok,

            # 3 * 6 = 18
            LMns_tok,
            LMes_tok,
            LVs_tok,
            LMis_tok,
            LMas_tok,
            LMcs_tok,
            SMns_tok,
            SMes_tok,
            SVs_tok,
            SMis_tok,
            SMas_tok,
            SMcs_tok,
            LnMns_tok,
            LnMes_tok,
            LnVs_tok,
            LnMis_tok,
            LnMas_tok,
            LnMcs_tok,

            SEs_tok,

            # 1 + 5 + 3 = 9
            LStsr_tok,
            SMnsr_tok,
            SMesr_tok,
            SStsr_tok,
            SMisr_tok,
            SMasr_tok,
            LnMnsr_tok,
            LnStsr_tok,
            LnMasr_tok,

            SEsr_tok,

            # 6
            SdLnMn_tok,
            SdLnMe_tok,
            SdLnSt_tok,
            SdLnV_tok,
            SdLnMi_tok,
            SdLnMa_tok,

            # 7
            SxLMn_tok,
            SxLMe_tok,
            SxLSt_tok,
            SxLV_tok,
            SxLMi_tok,
            SxLMa_tok,
            SxLMc_tok,

            # 6
            SxLMns_tok,
            SxLMes_tok,
            SxLVs_tok,
            SxLMis_tok,
            SxLMas_tok,
            SxLMcs_tok,

            # 7
            LdSxLMn_tok,
            LdSxLMe_tok,
            LdSxLSt_tok,
            LdSxLV_tok,
            LdSxLMi_tok,
            LdSxLMa_tok,
            LdSxLMc_tok,

            # 6
            SqLnMn_tok,
            SqLnMe_tok,
            SqLnSt_tok,
            SqLnV_tok,
            SqLnMi_tok,
            SqLnMa_tok,

            # 6
            LnqSMn_tok,
            LnqSMe_tok,
            LnqSSt_tok,
            LnqSV_tok,
            LnqSMi_tok,
            LnqSMa_tok,

            LMadMn_tok,
            SMadMn_tok,
            LnMadMn_tok,

            # 5
            SdLndx_tok,
            SxLdx_tok,
            LdSxLdx_tok,
            LnqSdx_tok,
            Ldx_tok,

            VSxL_tok,
            VSdLn_tok,

            VSxLs_tok,
            VSdLns_tok,

            VSxLsr_tok,
            VSdLnsr_tok,

            MadMnL_tok,
            MadMnLs_tok,
            MadMnLsr_tok,

            MadMnS_tok,
            MadMnSs_tok,
            MadMnSsr_tok,

            MadMnLn_tok,
            MadMnLns_tok,
            MadMnLnsr_tok,

            logits_scores_tok,
            softmax_scores_tok,
            norm_logits_scores_tok,
        )

        # 3 * 7 = 21
        self.LMn_tok = LMn_tok
        self.LMe_tok = LMe_tok
        self.LSt_tok = LSt_tok
        self.LV_tok = LV_tok
        self.LMi_tok = LMi_tok
        self.LMa_tok = LMa_tok
        self.LMc_tok = LMc_tok
        self.SMn_tok = SMn_tok
        self.SMe_tok = SMe_tok
        self.SSt_tok = SSt_tok
        self.SV_tok = SV_tok
        self.SMi_tok = SMi_tok
        self.SMa_tok = SMa_tok
        self.SMc_tok = SMc_tok
        self.LnMn_tok = LnMn_tok
        self.LnMe_tok = LnMe_tok
        self.LnSt_tok = LnSt_tok
        self.LnV_tok = LnV_tok
        self.LnMi_tok = LnMi_tok
        self.LnMa_tok = LnMa_tok
        self.LnMc_tok = LnMc_tok

        self.SE_tok = SE_tok

        # 3 * 6 = 18
        self.LMns_tok = LMns_tok
        self.LMes_tok = LMes_tok
        self.LVs_tok = LVs_tok
        self.LMis_tok = LMis_tok
        self.LMas_tok = LMas_tok
        self.LMcs_tok = LMcs_tok
        self.SMns_tok = SMns_tok
        self.SMes_tok = SMes_tok
        self.SVs_tok = SVs_tok
        self.SMis_tok = SMis_tok
        self.SMas_tok = SMas_tok
        self.SMcs_tok = SMcs_tok
        self.LnMns_tok = LnMns_tok
        self.LnMes_tok = LnMes_tok
        self.LnVs_tok = LnVs_tok
        self.LnMis_tok = LnMis_tok
        self.LnMas_tok = LnMas_tok
        self.LnMcs_tok = LnMcs_tok

        self.SEs_tok = SEs_tok

        # 1 + 5 + 3 = 9
        self.LStsr_tok = LStsr_tok
        self.SMnsr_tok = SMnsr_tok
        self.SMesr_tok = SMesr_tok
        self.SStsr_tok = SStsr_tok
        self.SMisr_tok = SMisr_tok
        self.SMasr_tok = SMasr_tok
        self.LnMnsr_tok = LnMnsr_tok
        self.LnStsr_tok = LnStsr_tok
        self.LnMasr_tok = LnMasr_tok

        self.SEsr_tok = SEsr_tok

        # 6
        self.SdLnMn_tok = SdLnMn_tok
        self.SdLnMe_tok = SdLnMe_tok
        self.SdLnSt_tok = SdLnSt_tok
        self.SdLnV_tok = SdLnV_tok
        self.SdLnMi_tok = SdLnMi_tok
        self.SdLnMa_tok = SdLnMa_tok

        # 7
        self.SxLMn_tok = SxLMn_tok
        self.SxLMe_tok = SxLMe_tok
        self.SxLSt_tok = SxLSt_tok
        self.SxLV_tok = SxLV_tok
        self.SxLMi_tok = SxLMi_tok
        self.SxLMa_tok = SxLMa_tok
        self.SxLMc_tok = SxLMc_tok

        # 6
        self.SxLMns_tok = SxLMns_tok
        self.SxLMes_tok = SxLMes_tok
        self.SxLVs_tok = SxLVs_tok
        self.SxLMis_tok = SxLMis_tok
        self.SxLMas_tok = SxLMas_tok
        self.SxLMcs_tok = SxLMcs_tok

        # 7
        self.LdSxLMn_tok = LdSxLMn_tok
        self.LdSxLMe_tok = LdSxLMe_tok
        self.LdSxLSt_tok = LdSxLSt_tok
        self.LdSxLV_tok = LdSxLV_tok
        self.LdSxLMi_tok = LdSxLMi_tok
        self.LdSxLMa_tok = LdSxLMa_tok
        self.LdSxLMc_tok = LdSxLMc_tok

        # 6
        self.SqLnMn_tok = SqLnMn_tok
        self.SqLnMe_tok = SqLnMe_tok
        self.SqLnSt_tok = SqLnSt_tok
        self.SqLnV_tok = SqLnV_tok
        self.SqLnMi_tok = SqLnMi_tok
        self.SqLnMa_tok = SqLnMa_tok

        # 6
        self.LnqSMn_tok = LnqSMn_tok
        self.LnqSMe_tok = LnqSMe_tok
        self.LnqSSt_tok = LnqSSt_tok
        self.LnqSV_tok = LnqSV_tok
        self.LnqSMi_tok = LnqSMi_tok
        self.LnqSMa_tok = LnqSMa_tok

        self.LMadMn_tok = LMadMn_tok
        self.SMadMn_tok = SMadMn_tok
        self.LnMadMn_tok = LnMadMn_tok

        # 5
        self.SdLndx_tok = SdLndx_tok
        self.SxLdx_tok = SxLdx_tok
        self.LdSxLdx_tok = LdSxLdx_tok
        self.LnqSdx_tok = LnqSdx_tok
        self.Ldx_tok = Ldx_tok

        self.VSxL_tok = VSxL_tok
        self.VSdLn_tok = VSdLn_tok

        self.VSxLs_tok = VSxLs_tok
        self.VSdLns_tok = VSdLns_tok

        self.VSxLsr_tok = VSxLsr_tok
        self.VSdLnsr_tok = VSdLnsr_tok

        self.MadMnL_tok = MadMnL_tok
        self.MadMnLs_tok = MadMnLs_tok
        self.MadMnLsr_tok = MadMnLsr_tok

        self.MadMnS_tok = MadMnS_tok
        self.MadMnSs_tok = MadMnSs_tok
        self.MadMnSsr_tok = MadMnSsr_tok
    
        self.MadMnLn_tok = MadMnLn_tok
        self.MadMnLns_tok = MadMnLns_tok
        self.MadMnLnsr_tok = MadMnLnsr_tok

        self.logits_scores_tok = logits_scores_tok
        self.softmax_scores_tok = softmax_scores_tok
        self.norm_logits_scores_tok = norm_logits_scores_tok

    def update_stats(self, batch_idx, current_seq_batch, gen_tokens, logits_tok, softmax_tok, norm_logits_tok, vocab_size):
        """
        Update all statistics and metrics on the provided logits and softMa scores.
        This function encapsulates all the code that calculates stats and writes them into the initialized tensors.
        """
        # Adjust scores shape to accomodate vocab_size
        if self.logits_scores_tok.shape[2] != vocab_size:
            scores_shape = list(self.logits_scores_tok.shape)
            scores_shape[2] = vocab_size

            self.logits_scores_tok = self.logits_scores_tok.resize_(*scores_shape)
            self.softmax_scores_tok = self.softmax_scores_tok.resize_(*scores_shape)
            self.norm_logits_scores_tok = self.norm_logits_scores_tok.resize_(*scores_shape)

        # Handles Vying length of generated tokens while maintaining contiguity
        # 7 X 3 = 21 measures
        self.LMn_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.nanmean(logits_tok[batch_idx], dim=-1) # Shape: (batch_size, :gen_tokens, vocab_size)
        self.LMe_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.median(torch.sort(logits_tok[batch_idx], dim=-1, descending=False).values, dim=-1).values
        self.LSt_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.std(logits_tok[batch_idx], dim=-1)
        self.LV_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.var(logits_tok[batch_idx], dim=-1, unbiased=False)
        self.LMi_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.min(logits_tok[batch_idx], dim=-1).values
        self.LMa_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.max(logits_tok[batch_idx], dim=-1).values
        self.LMc_tok[batch_idx, current_seq_batch, :gen_tokens] = mcsi(logits_tok[batch_idx], dim=-1)
        self.SMn_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.nanmean(softmax_tok[batch_idx], dim=-1) # Shape: (batch_size, :gen_tokens, vocab_size)
        self.SMe_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.median(torch.sort(softmax_tok[batch_idx], dim=-1, descending=False).values, dim=-1).values
        self.SSt_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.std(softmax_tok[batch_idx], dim=-1)
        self.SV_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.var(softmax_tok[batch_idx], dim=-1, unbiased=False)
        self.SMi_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.min(softmax_tok[batch_idx], dim=-1).values
        self.SMa_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.max(softmax_tok[batch_idx], dim=-1).values
        self.SMc_tok[batch_idx, current_seq_batch, :gen_tokens] = mcsi(logits_tok[batch_idx], dim=-1)
        self.LnMn_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.nanmean(norm_logits_tok[batch_idx], dim=-1) # Shape: (batch_size, :gen_tokens, vocab_size)
        self.LnMe_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.median(torch.sort(norm_logits_tok[batch_idx], dim=-1, descending=False).values, dim=-1).values
        self.LnSt_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.std(norm_logits_tok[batch_idx], dim=-1)
        self.LnV_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.var(norm_logits_tok[batch_idx], dim=-1, unbiased=False)
        self.LnMi_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.min(norm_logits_tok[batch_idx], dim=-1).values
        self.LnMa_tok[batch_idx, current_seq_batch, :gen_tokens] = torch.max(norm_logits_tok[batch_idx], dim=-1).values
        self.LnMc_tok[batch_idx, current_seq_batch, :gen_tokens] = mcsi(norm_logits_tok[batch_idx], dim=-1)

        self.SE_tok[batch_idx, current_seq_batch, :gen_tokens] = -torch.sum(softmax_tok[batch_idx] * torch.log(softmax_tok[batch_idx]), dim=-1)
        
        self.LMn_tok[batch_idx, current_seq_batch] = self.LMn_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LMe_tok[batch_idx, current_seq_batch] = self.LMe_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LSt_tok[batch_idx, current_seq_batch] = self.LSt_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LV_tok[batch_idx, current_seq_batch] = self.LV_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LMi_tok[batch_idx, current_seq_batch] = self.LMi_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LMa_tok[batch_idx, current_seq_batch] = self.LMa_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LMc_tok[batch_idx, current_seq_batch] = self.LMc_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SMn_tok[batch_idx, current_seq_batch] = self.SMn_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SMe_tok[batch_idx, current_seq_batch] = self.SMe_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SSt_tok[batch_idx, current_seq_batch] = self.SSt_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SV_tok[batch_idx, current_seq_batch] = self.SV_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SMi_tok[batch_idx, current_seq_batch] = self.SMi_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SMa_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.SMc_tok[batch_idx, current_seq_batch] = self.SMc_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnMn_tok[batch_idx, current_seq_batch] = self.LnMn_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnMe_tok[batch_idx, current_seq_batch] = self.LnMe_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnSt_tok[batch_idx, current_seq_batch] = self.LnSt_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnV_tok[batch_idx, current_seq_batch] = self.LnV_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnMi_tok[batch_idx, current_seq_batch] = self.LnMi_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnMa_tok[batch_idx, current_seq_batch] = self.LnMa_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)
        self.LnMc_tok[batch_idx, current_seq_batch] = self.LnMc_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)

        self.SE_tok[batch_idx, current_seq_batch] = self.SE_tok[batch_idx, current_seq_batch].nan_to_num(nan=1e-9)

        # 3 * 6 = 18
        self.LMns_tok[batch_idx, current_seq_batch] = self.LMn_tok[batch_idx, current_seq_batch] ** 2
        self.LMes_tok[batch_idx, current_seq_batch] = self.LMe_tok[batch_idx, current_seq_batch] ** 2
        self.LVs_tok[batch_idx, current_seq_batch] = self.LV_tok[batch_idx, current_seq_batch] ** 2
        self.LMis_tok[batch_idx, current_seq_batch] = self.LMi_tok[batch_idx, current_seq_batch] ** 2
        self.LMas_tok[batch_idx, current_seq_batch] = self.LMa_tok[batch_idx, current_seq_batch] ** 2
        self.LMcs_tok[batch_idx, current_seq_batch] = self.LMc_tok[batch_idx, current_seq_batch] ** 2
        self.SMns_tok[batch_idx, current_seq_batch] = self.SMn_tok[batch_idx, current_seq_batch] ** 2
        self.SMes_tok[batch_idx, current_seq_batch] = self.SMe_tok[batch_idx, current_seq_batch] ** 2
        self.SVs_tok[batch_idx, current_seq_batch] = self.SV_tok[batch_idx, current_seq_batch] ** 2
        self.SMis_tok[batch_idx, current_seq_batch] = self.SMi_tok[batch_idx, current_seq_batch] ** 2
        self.SMas_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch] ** 2
        self.SMcs_tok[batch_idx, current_seq_batch] = self.SMc_tok[batch_idx, current_seq_batch] ** 2
        self.LnMns_tok[batch_idx, current_seq_batch] = self.LnMn_tok[batch_idx, current_seq_batch] ** 2
        self.LnMes_tok[batch_idx, current_seq_batch] = self.LnMe_tok[batch_idx, current_seq_batch] ** 2
        self.LnVs_tok[batch_idx, current_seq_batch] = self.LnV_tok[batch_idx, current_seq_batch] ** 2
        self.LnMis_tok[batch_idx, current_seq_batch] = self.LnMi_tok[batch_idx, current_seq_batch] ** 2
        self.LnMas_tok[batch_idx, current_seq_batch] = self.LnMa_tok[batch_idx, current_seq_batch] ** 2
        self.LnMcs_tok[batch_idx, current_seq_batch] = self.LnMc_tok[batch_idx, current_seq_batch] ** 2

        self.SEs_tok[batch_idx, current_seq_batch] = self.SE_tok[batch_idx, current_seq_batch] ** 2

        # 1 + 5 + 3 = 9
        self.LStsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.LSt_tok[batch_idx, current_seq_batch])
        self.SMnsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SMn_tok[batch_idx, current_seq_batch])
        self.SMesr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SMe_tok[batch_idx, current_seq_batch])
        self.SStsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SSt_tok[batch_idx, current_seq_batch])
        self.SMisr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SMi_tok[batch_idx, current_seq_batch])
        self.SMasr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SMa_tok[batch_idx, current_seq_batch])
        self.LnMnsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.LnMn_tok[batch_idx, current_seq_batch])
        self.LnStsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.LnSt_tok[batch_idx, current_seq_batch])
        self.LnMasr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.LnMa_tok[batch_idx, current_seq_batch])

        self.SEsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.SE_tok[batch_idx, current_seq_batch])
        
        # 6
        self.SdLnMn_tok[batch_idx, current_seq_batch] = self.SMn_tok[batch_idx, current_seq_batch] - self.LnMn_tok[batch_idx, current_seq_batch]
        self.SdLnMe_tok[batch_idx, current_seq_batch] = self.SMe_tok[batch_idx, current_seq_batch] - self.LnMe_tok[batch_idx, current_seq_batch]
        self.SdLnSt_tok[batch_idx, current_seq_batch] = self.SSt_tok[batch_idx, current_seq_batch] - self.LnSt_tok[batch_idx, current_seq_batch]
        self.SdLnV_tok[batch_idx, current_seq_batch] = self.SV_tok[batch_idx, current_seq_batch] - self.LnV_tok[batch_idx, current_seq_batch]
        self.SdLnMi_tok[batch_idx, current_seq_batch] = self.SMi_tok[batch_idx, current_seq_batch] - self.LnMi_tok[batch_idx, current_seq_batch]
        self.SdLnMa_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch] - self.LnMa_tok[batch_idx, current_seq_batch]

        # 7
        self.SxLMn_tok[batch_idx, current_seq_batch] = self.SMn_tok[batch_idx, current_seq_batch] * self.LMn_tok[batch_idx, current_seq_batch]
        self.SxLMe_tok[batch_idx, current_seq_batch] = self.SMe_tok[batch_idx, current_seq_batch] * self.LMe_tok[batch_idx, current_seq_batch]
        self.SxLSt_tok[batch_idx, current_seq_batch] = self.SSt_tok[batch_idx, current_seq_batch] * self.LSt_tok[batch_idx, current_seq_batch]
        self.SxLV_tok[batch_idx, current_seq_batch] = self.SV_tok[batch_idx, current_seq_batch] * self.LV_tok[batch_idx, current_seq_batch]
        self.SxLMi_tok[batch_idx, current_seq_batch] = self.SMi_tok[batch_idx, current_seq_batch] * self.LMi_tok[batch_idx, current_seq_batch]
        self.SxLMa_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch] * self.LMa_tok[batch_idx, current_seq_batch]
        self.SxLMc_tok[batch_idx, current_seq_batch] = self.SMc_tok[batch_idx, current_seq_batch] * self.LMc_tok[batch_idx, current_seq_batch]

        # 6
        self.SxLMns_tok[batch_idx, current_seq_batch] = (self.SMn_tok[batch_idx, current_seq_batch] * self.LMn_tok[batch_idx, current_seq_batch]) ** 2
        self.SxLMes_tok[batch_idx, current_seq_batch] = (self.SMe_tok[batch_idx, current_seq_batch] * self.LMe_tok[batch_idx, current_seq_batch]) ** 2
        self.SxLVs_tok[batch_idx, current_seq_batch] = (self.SV_tok[batch_idx, current_seq_batch] * self.LV_tok[batch_idx, current_seq_batch]) ** 2
        self.SxLMis_tok[batch_idx, current_seq_batch] = (self.SMi_tok[batch_idx, current_seq_batch] * self.LMi_tok[batch_idx, current_seq_batch]) ** 2
        self.SxLMas_tok[batch_idx, current_seq_batch] = (self.SMa_tok[batch_idx, current_seq_batch] * self.LMa_tok[batch_idx, current_seq_batch]) ** 2
        self.SxLMcs_tok[batch_idx, current_seq_batch] = (self.SMc_tok[batch_idx, current_seq_batch] * self.LMc_tok[batch_idx, current_seq_batch]) ** 2

        # 7
        self.LdSxLMn_tok[batch_idx, current_seq_batch] = self.LMn_tok[batch_idx, current_seq_batch] - self.SxLMn_tok[batch_idx, current_seq_batch]
        self.LdSxLMe_tok[batch_idx, current_seq_batch] = self.LMe_tok[batch_idx, current_seq_batch] - self.SxLMe_tok[batch_idx, current_seq_batch]
        self.LdSxLSt_tok[batch_idx, current_seq_batch] = self.LSt_tok[batch_idx, current_seq_batch] - self.SxLSt_tok[batch_idx, current_seq_batch]
        self.LdSxLV_tok[batch_idx, current_seq_batch] = self.LV_tok[batch_idx, current_seq_batch] - self.SxLV_tok[batch_idx, current_seq_batch]
        self.LdSxLMi_tok[batch_idx, current_seq_batch] = self.LMi_tok[batch_idx, current_seq_batch] - self.SxLMi_tok[batch_idx, current_seq_batch]
        self.LdSxLMa_tok[batch_idx, current_seq_batch] = self.LMa_tok[batch_idx, current_seq_batch] - self.SxLMa_tok[batch_idx, current_seq_batch]
        self.LdSxLMc_tok[batch_idx, current_seq_batch] = self.LMc_tok[batch_idx, current_seq_batch] - self.SxLMc_tok[batch_idx, current_seq_batch]

        # 6
        self.SqLnMn_tok[batch_idx, current_seq_batch] = self.SMn_tok[batch_idx, current_seq_batch] / self.LnMn_tok[batch_idx, current_seq_batch]
        self.SqLnMe_tok[batch_idx, current_seq_batch] = self.SMe_tok[batch_idx, current_seq_batch] / self.LnMe_tok[batch_idx, current_seq_batch]
        self.SqLnSt_tok[batch_idx, current_seq_batch] = self.SSt_tok[batch_idx, current_seq_batch] / self.LnSt_tok[batch_idx, current_seq_batch]
        self.SqLnV_tok[batch_idx, current_seq_batch] = self.SV_tok[batch_idx, current_seq_batch] / self.LnV_tok[batch_idx, current_seq_batch]
        self.SqLnMi_tok[batch_idx, current_seq_batch] = self.SMi_tok[batch_idx, current_seq_batch] / self.LnMi_tok[batch_idx, current_seq_batch]
        self.SqLnMa_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch] / self.LnMa_tok[batch_idx, current_seq_batch]

        # 6
        self.LnqSMn_tok[batch_idx, current_seq_batch] = self.LnMn_tok[batch_idx, current_seq_batch] / self.SMn_tok[batch_idx, current_seq_batch]
        self.LnqSMe_tok[batch_idx, current_seq_batch] = self.LnMe_tok[batch_idx, current_seq_batch] / self.SMe_tok[batch_idx, current_seq_batch]
        self.LnqSSt_tok[batch_idx, current_seq_batch] = self.LnSt_tok[batch_idx, current_seq_batch] / self.SSt_tok[batch_idx, current_seq_batch]
        self.LnqSV_tok[batch_idx, current_seq_batch] = self.LnV_tok[batch_idx, current_seq_batch] / self.SV_tok[batch_idx, current_seq_batch]
        self.LnqSMi_tok[batch_idx, current_seq_batch] = self.LnMi_tok[batch_idx, current_seq_batch] / self.SMi_tok[batch_idx, current_seq_batch]
        self.LnqSMa_tok[batch_idx, current_seq_batch] = self.LnMa_tok[batch_idx, current_seq_batch] / self.SMa_tok[batch_idx, current_seq_batch]

        self.LMadMn_tok[batch_idx, current_seq_batch] = self.LMa_tok[batch_idx, current_seq_batch] - self.LMn_tok[batch_idx, current_seq_batch]
        self.SMadMn_tok[batch_idx, current_seq_batch] = self.SMa_tok[batch_idx, current_seq_batch] - self.SMn_tok[batch_idx, current_seq_batch]
        self.LnMadMn_tok[batch_idx, current_seq_batch] = self.LnMa_tok[batch_idx, current_seq_batch] - self.LnMn_tok[batch_idx, current_seq_batch]

        self.SdLndx_tok[batch_idx, current_seq_batch] = torch.trapz(softmax_tok[batch_idx] - norm_logits_tok[batch_idx])
        self.SxLdx_tok[batch_idx, current_seq_batch] = torch.trapz(softmax_tok[batch_idx] * logits_tok[batch_idx])
        self.LdSxLdx_tok[batch_idx, current_seq_batch] = torch.trapz(logits_tok[batch_idx] - softmax_tok[batch_idx] * logits_tok[batch_idx])
        self.LnqSdx_tok[batch_idx, current_seq_batch] = torch.trapz(norm_logits_tok[batch_idx] / softmax_tok[batch_idx])
        self.Ldx_tok[batch_idx, current_seq_batch] = torch.trapz(logits_tok[batch_idx])

        self.VSxL_tok[batch_idx, current_seq_batch] = torch.var(softmax_tok[batch_idx] * logits_tok[batch_idx], dim=-1)
        self.VSdLn_tok[batch_idx, current_seq_batch] = torch.var(softmax_tok[batch_idx] - norm_logits_tok[batch_idx], dim=-1)

        self.VSxLs_tok[batch_idx, current_seq_batch] = self.VSxL_tok[batch_idx, current_seq_batch] ** 2
        self.VSdLns_tok[batch_idx, current_seq_batch] = self.VSdLn_tok[batch_idx, current_seq_batch] ** 2

        self.VSxLsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.VSxL_tok[batch_idx, current_seq_batch])
        self.VSdLnsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.VSdLn_tok[batch_idx, current_seq_batch])

        self.MadMnL_tok[batch_idx, current_seq_batch] =  torch.max(logits_tok[batch_idx] * logits_tok[batch_idx], dim=-1).values - torch.nanmean(logits_tok[batch_idx] * logits_tok[batch_idx], dim=-1)
        self.MadMnLs_tok[batch_idx, current_seq_batch] = self.MadMnL_tok[batch_idx, current_seq_batch] ** 2
        self.MadMnLsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.MadMnL_tok[batch_idx, current_seq_batch])

        self.MadMnS_tok[batch_idx, current_seq_batch] =  torch.max(softmax_tok[batch_idx] * logits_tok[batch_idx], dim=-1).values - torch.nanmean(softmax_tok[batch_idx] * logits_tok[batch_idx], dim=-1)
        self.MadMnSs_tok[batch_idx, current_seq_batch] = self.MadMnS_tok[batch_idx, current_seq_batch] ** 2
        self.MadMnSsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.MadMnS_tok[batch_idx, current_seq_batch])

        self.MadMnLn_tok[batch_idx, current_seq_batch] =  torch.max(norm_logits_tok[batch_idx] * norm_logits_tok[batch_idx], dim=-1).values - torch.nanmean(norm_logits_tok[batch_idx] * norm_logits_tok[batch_idx], dim=-1)
        self.MadMnLns_tok[batch_idx, current_seq_batch] = self.MadMnLn_tok[batch_idx, current_seq_batch] ** 2
        self.MadMnLnsr_tok[batch_idx, current_seq_batch] = torch.sqrt(self.MadMnLn_tok[batch_idx, current_seq_batch])

        self.logits_scores_tok[batch_idx, current_seq_batch] = logits_tok[batch_idx]
        self.softmax_scores_tok[batch_idx, current_seq_batch] = softmax_tok[batch_idx]
        self.norm_logits_scores_tok[batch_idx, current_seq_batch] = norm_logits_tok[batch_idx]

###############################################
# Result Saving
###############################################

class ResultSaver:
    @staticmethod
    def data_to_h5(match_or_mismatch_total, tok_data, main_path):
        with h5py.File(main_path, "w") as f:
            match_or_mismatch_total_bytes = [str(s).encode('utf-8') for s in match_or_mismatch_total]
            f.create_dataset("match_or_mismatch_total", data=match_or_mismatch_total_bytes)
            for key, value in tok_data.items():
                f.create_dataset(key, data=value)

    @staticmethod
    def scores_to_h5(tok_scores, second_path):
        with h5py.File(second_path, "w") as f:
            for key, value in tok_scores.items():
                f.create_dataset(key, data=value)
    
    @staticmethod
    def answers_to_h5(match_or_mismatch_total, main_path):
        with h5py.File(main_path, "w") as f:
            match_or_mismatch_total_bytes = [str(s).encode('utf-8') for s in match_or_mismatch_total]
            f.create_dataset("match_or_mismatch_total", data=match_or_mismatch_total_bytes)

###############################################
# Main Inference Procedure
###############################################

def inference(config: Config):
    data_preparer = DataPreparer(config)
    prompt_gen = PromptGenerator(data_preparer, config)
    inference_engine = ModelInference(config)
    match_or_mismatch_total = []

    # Flattened answers container will be filled after test prompts are fetched below
    for epoch_idx in range(config.num_epochs):
        print(f"Starting epoch {epoch_idx + 1}/{config.num_epochs}\n")
        start_time = time.time()

        full_prompts, answers = prompt_gen.concatenate(epoch_idx)

        if config.full_mmlu_override:
            total_seq = len(answers)
            total_seq_batches = (total_seq + config.batch_size - 1) // config.batch_size
            stats_collector = StatsCollector(config, total_seq_batches=total_seq_batches)
        elif not config.full_mmlu_override and epoch_idx == 0:
            total_seq = config.total_seq
            total_seq_batches = config.total_seq_batches
            stats_collector = StatsCollector(config) # Holds and updates all metrics

        # Run inference and stats update
        inference_engine.run_inference(
            full_prompts, 
            epoch_idx, 
            answers, 
            match_or_mismatch_total, 
            stats_collector=stats_collector,
            total_seq_batches=total_seq_batches
            )
        
        end_time = time.time()
        print(f"Epoch {epoch_idx + 1} processing time: {end_time - start_time:.2f}")
    
    #print(stats_collector.mean_logits_mean_seq)
    if config.save_stats:
        print("Storing data...")

        ###########################
        # Token Data
        ###########################

        tok_data = {}
        tok_scores = {}

        # 3 * 7 = 21
        tok_data["LMn"] = stats_collector.LMn_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMe"] = stats_collector.LMe_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LSt"] = stats_collector.LSt_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LV"] = stats_collector.LV_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMi"] = stats_collector.LMi_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMa"] = stats_collector.LMa_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMc"] = stats_collector.LMc_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMn"] = stats_collector.SMn_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMe"] = stats_collector.SMe_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SSt"] = stats_collector.SSt_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SV"] = stats_collector.SV_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMi"] = stats_collector.SMi_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMa"] = stats_collector.SMa_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMc"] = stats_collector.SMc_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMn"] = stats_collector.LnMn_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMe"] = stats_collector.LnMe_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnSt"] = stats_collector.LnSt_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnV"] = stats_collector.LnV_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMi"] = stats_collector.LnMi_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMa"] = stats_collector.LnMa_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMc"] = stats_collector.LnMc_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()

        tok_data["SE"] = stats_collector.SE_tok.squeeze(-1).reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SN"] = -tok_data["SE"]

        # 3 * 6 = 21
        tok_data["LMns"] = stats_collector.LMns_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMes"] = stats_collector.LMes_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LVs"] = stats_collector.LVs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMis"] = stats_collector.LMis_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMas"] = stats_collector.LMas_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LMcs"] = stats_collector.LMcs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMns"] = stats_collector.SMns_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMes"] = stats_collector.SMes_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SVs"] = stats_collector.SVs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMis"] = stats_collector.SMis_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMas"] = stats_collector.SMas_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMcs"] = stats_collector.SMcs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMns"] = stats_collector.LnMns_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMes"] = stats_collector.LnMes_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnVs"] = stats_collector.LnVs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMis"] = stats_collector.LnMis_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMas"] = stats_collector.LnMas_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMcs"] = stats_collector.LnMcs_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["SEs"] = stats_collector.SEs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SNs"] = -tok_data["SEs"]

        # 1 + 5 + 3 = 9
        tok_data["LStsr"] = stats_collector.LStsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMnsr"] = stats_collector.SMnsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMesr"] = stats_collector.SMesr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SStsr"] = stats_collector.SStsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMisr"] = stats_collector.SMisr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMasr"] = stats_collector.SMasr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMnsr"] = stats_collector.LnMnsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnStsr"] = stats_collector.LnStsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMasr"] = stats_collector.LnMasr_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["SEsr"] = stats_collector.SEsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SNsr"] = -tok_data["SEsr"]

        # 6
        tok_data["SdLnMn"] = stats_collector.SdLnMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SdLnMe"] = stats_collector.SdLnMe_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SdLnSt"] = stats_collector.SdLnSt_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SdLnV"] = stats_collector.SdLnV_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SdLnMi"] = stats_collector.SdLnMi_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SdLnMa"] = stats_collector.SdLnMa_tok.reshape(-1)[:total_seq].cpu().numpy()

        # 7
        tok_data["SxLMn"] = stats_collector.SxLMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMe"] = stats_collector.SxLMe_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLSt"] = stats_collector.SxLSt_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLV"] = stats_collector.SxLV_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMi"] = stats_collector.SxLMi_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMa"] = stats_collector.SxLMa_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMc"] = stats_collector.SxLMc_tok.reshape(-1)[:total_seq].cpu().numpy()

        # 6
        tok_data["SxLMns"] = stats_collector.SxLMns_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMes"] = stats_collector.SxLMes_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLVs"] = stats_collector.SxLVs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMis"] = stats_collector.SxLMis_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMas"] = stats_collector.SxLMas_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLMcs"] = stats_collector.SxLMcs_tok.reshape(-1)[:total_seq].cpu().numpy()

        # 7
        tok_data["LdSxLMn"] = stats_collector.LdSxLMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLMe"] = stats_collector.LdSxLMe_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLSt"] = stats_collector.LdSxLSt_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLV"] = stats_collector.LdSxLV_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLMi"] = stats_collector.LdSxLMi_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLMa"] = stats_collector.LdSxLMa_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLMc"] = stats_collector.LdSxLMc_tok.reshape(-1)[:total_seq].cpu().numpy()

        # 6
        tok_data["SqLnMn"] = stats_collector.SqLnMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SqLnMe"] = stats_collector.SqLnMe_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SqLnSt"] = stats_collector.SqLnSt_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SqLnV"] = stats_collector.SqLnV_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SqLnMi"] = stats_collector.SqLnMi_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SqLnMa"] = stats_collector.SqLnMa_tok.reshape(-1)[:total_seq].cpu().numpy()

        # 6
        tok_data["LnqSMn"] = stats_collector.LnqSMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSMe"] = stats_collector.LnqSMe_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSSt"] = stats_collector.LnqSSt_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSV"] = stats_collector.LnqSV_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSMi"] = stats_collector.LnqSMi_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSMa"] = stats_collector.LnqSMa_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["LMadMn"] = stats_collector.LMadMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SMadMn"] = stats_collector.SMadMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnMadMn"] = stats_collector.LnMadMn_tok.reshape(-1)[:total_seq].cpu().numpy()
        
        # 5
        tok_data["SdLndx"] = stats_collector.SdLndx_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["SxLdx"] = stats_collector.SxLdx_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LdSxLdx"] = stats_collector.LdSxLdx_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["LnqSdx"] = stats_collector.LnqSdx_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["Ldx"] = stats_collector.Ldx_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["VSxL"] = stats_collector.VSxL_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["VSdLn"] = stats_collector.VSdLn_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["VSxLs"] = stats_collector.VSxLs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["VSdLns"] = stats_collector.VSdLns_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["VSxLsr"] = stats_collector.VSxLsr_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["VSdLnsr"] = stats_collector.VSdLnsr_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["MadMnS"] = stats_collector.MadMnS_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnSs"] = stats_collector.MadMnSs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnSsr"] = stats_collector.MadMnSsr_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["MadMnL"] = stats_collector.MadMnL_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnLs"] = stats_collector.MadMnLs_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnLsr"] = stats_collector.MadMnLsr_tok.reshape(-1)[:total_seq].cpu().numpy()

        tok_data["MadMnLn"] = stats_collector.MadMnLn_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnLns"] = stats_collector.MadMnLns_tok.reshape(-1)[:total_seq].cpu().numpy()
        tok_data["MadMnLnsr"] = stats_collector.MadMnLnsr_tok.reshape(-1)[:total_seq].cpu().numpy()

        def _min_max_normalization(array):
            return (array - array.min()) / (array.max() - array.min())

        tok_data["SxLnVspSNsr"] = _min_max_normalization(tok_data["VSxLs"]) + _min_max_normalization(tok_data["SNsr"])
        tok_data["SxLnVspSNsrpLMadMn"] = tok_data["SxLnVspSNsr"] + _min_max_normalization(tok_data["LMadMn"])
        tok_data["SxLnVspSNsrpMadMnSs"] = tok_data["SxLnVspSNsr"] + _min_max_normalization(tok_data["MadMnSs"])
        tok_data["SVspSNsr"] = _min_max_normalization(tok_data["SVs"]) + _min_max_normalization(tok_data["SNsr"])

        # Check for NaNs and infs
        for key, value in tok_data.items():
            has_nan = np.isnan(value).any()
            has_inf = np.isinf(value).any()
            if has_nan:
                print(f'{key} has NaN')
            if has_inf:
                print(f'{key} has inf')

        ResultSaver.data_to_h5(match_or_mismatch_total, tok_data, config.main_path)

        ###########################
        # Scores Data
        ###########################
        vocab_size = stats_collector.logits_scores_tok.shape[2] # Extract vocab size for current model

        tok_scores['logits'] = stats_collector.logits_scores_tok.reshape(-1, vocab_size)[:total_seq].cpu().numpy()
        tok_scores['softmax'] = stats_collector.softmax_scores_tok.reshape(-1, vocab_size)[:total_seq].cpu().numpy()
        tok_scores['normalized logits'] = stats_collector.norm_logits_scores_tok.reshape(-1, vocab_size)[:total_seq].cpu().numpy()

        # Check for NaNs and infs
        for key, value in tok_scores.items():
            has_nan = np.isnan(value).any()
            has_inf = np.isinf(value).any()
            if has_nan:
                print(f'{key} has NaN')
            if has_inf:
                print(f'{key} has inf')

        ResultSaver.scores_to_h5(tok_scores, config.second_path)

        print('Storage completed.')
    
if __name__ == '__main__':
    config = Config()
    inference(config)
