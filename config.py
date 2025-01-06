import os
import torch

class Config:
    def __init__(self):
        # Parameter configuration
        self.num_epochs = 1
        self.num_shots = 5
        self.max_new_tokens = 1
        self.batch_size = 1
        """
        `seq_per_dataset`:
        iterates over the 57 datasets for the assigned integer.
        Example: `seq_per_dataset` = 1 returns 57 responses.

        `full_mmlu_override`:
        Setting to `True` overrides `seq_per_dataset` and runs
        the model for all 14,042 questions in MMLU.
        """
        self.seq_per_dataset = 1
        self.full_mmlu_override = True

        # Output configuration
        self.print_responses = True
        self.save_stats = True

        # Instruct models
        self.instruct = True

        # Model selection interface (only choose one)
        self.l1 = False # Llama 3.2 1B
        self.l3 = False  # Llama 3.2 3B
        self.l8 = True # Llama 3.1 8B
        self.f1 = False # Falcon3 1B
        self.f3 = False # Falcon3 3B
        self.f7 = False # Falcon3 7B
        self.f0 = False # Falcon3 10B
        self.o7 = True # olmo 2 7B
        self.m7 = False # Mistral v3 7B
        " Must `self.tool_call = True` only for falcon3 instruct family"
        self.tool_call = True if self.f1 or self.f3 or self.f7 or self.f0 else False
        

        # IDs for loading model
        self.model_ids = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "tiiuae/Falcon3-1B-Instruct",
            "tiiuae/Falcon3-3B-Instruct",
            "tiiuae/Falcon3-7B-Instruct",
            "tiiuae/Falcon3-10B-Instruct",
            "allenai/OLMo-2-1124-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            ]
        self.model_indices = [self.l1, self.l3, self.l8, self.f1, self.f3, self.f7, self.f0, self.o7, self.m7]

        # Model loading configuration
        self.model_id = self.model_ids[self.model_indices.index(True)] #'MBZUAI/LaMini-GPT-124M' #self.model_ids[self.model_indices.index(True)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Meta parameters
        self.num_datasets = 57 # Original MMLU benchmark has 57 csv files
        self.seq_per_epoch = self.seq_per_dataset * self.num_datasets
        self.total_seq = self.seq_per_epoch * self.num_epochs
        self.total_seq_batches = (self.total_seq + (self.batch_size - 1)) // self.batch_size
        self.precision = torch.float64 # For precise score data calculations

        # Filenames (which ar customizable of course) for loading saved output data 
        self.model_paths = [
            "llama-3.2-1b-instruct-mmlu-stats.h5",
            "llama-3.2-3b-instruct-mmlu-stats.h5",
            "llama-3.1-8b-instruct-mmlu-stats.h5",
            "falcon3-1b-instruct-mmlu-stats.h5",
            "falcon3-3b-instruct-mmlu-stats.h5",
            "falcon3-7b-instruct-mmlu-stats.h5",
            "falcon3-10b-instruct-mmlu-stats.h5",
            "olmo-2-1124-7b-instruct-mmlu-stats.h5",
            "mistral-7b-instruct-v0.3-mmlu-stats.h5"
            ]

        # Path to filenames
        self.model_path = self.model_paths[self.model_indices.index(True)]  
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.main_path = os.path.join(self.dir_path, self.model_path)

        # Path to saved scores data
        self.second_path = os.path.join(self.dir_path, "scores.h5")

        # Paths to MMLU csv files
        self.folder_path_val = '/home/ubuntu/venv/mmlu/val' #'C:/Users/charl/Documents/MMLU/data/val' #
        self.folder_path_test = '/home/ubuntu/venv/mmlu/test' # 'C:/Users/charl/Documents/MMLU/data/test'