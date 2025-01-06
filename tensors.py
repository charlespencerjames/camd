import torch

"""
###########################
# Naming convention
###########################

------- Scores -------
L = Logits
S = Softmax
Ln = Normalized Logits

------- Metrics -------
Mn = Mean
Me = Median
St = Standard Deviation
V = Variance
Ma = Maximum
Mi = Minimum
Mc = Mean-Centered Skewness Index

------- Operators -------
~~~ Binary ~~~
p = Addition (plus)
d = Subtraction (difference)
x = Multiplication (times)
q = Division (quotient)

~~~ Unary ~~~
s = Square
sr = Square Root
dx = integral

###########################
# Ordering Logic
###########################
------ Unit Composition ------
A unit comprises a set of at least one element from 'Scores' and 'Metrics', 
such as 'LMn' for 'logits mean', which are is the mean of logit scores.

------ Binary Vs. Unary Operators ------
~~~ Binary Operators ~~~ 
Binary operators must be placed in-between either two 'Scores', 'Metrics', or 'Units'.

Example:
    'SxL' is 'softmax scores times logits scores', and if 'V' was added on the end,
    it would be 'SxLV', which would calculate the variance of their product.

~~~ Unary Operators ~~~
Unary operators are placed at the end of 'Scores' or 'Metrics', and in the case they
are placed at the end of a 'Unit', they refer to the nearest score or metric to their left.

Example:
    'LnMas' is 'normalized logits maximum squared' and 'SStsr' is the 'square root of the
    softmax probabilities standard deviation'.

    'Ldx' is the 'integral of the logit scores,' which is a rare case when a unary operator
    is employed directly on 'Scores' instead of 'Metrics'

------ Intra-Unit Operations ------
Operations can take place intra-unit where the operator is placed between
either two 'Scores' elements or two 'Metrics' elements. 

Examples:
    When the operator is between metrics in the unit (an intra-metric operation),
    'LnMadMn' means 'normalized logits maximum minus normalized logits mean'.

    When an operator is between two scores in the unit (an intra-scores operation),
    'SqLnSt' means 'softmax standard deviation divided by normalized logits standard
    deviation'.

    There can be further combinations as well, such as 'LdSxLV', and so on.

------ Inter-Unit Operations ------
Operations can take place inter-unit as well where the operator is placed between
two complete units.

Examples:
    Any type of unit can be subject to inter-unit operations. 'SxLnVspSNsr' means
    'the squared variance of the product of softmax and normalized logits scores pluse
    the square root of the negentropy of the softmax scores.'

    For another example, 'SxLnVspSNsrpMadMnSs' means 'the squared variance of the product
    of softmax and normalized logit scores plus the square root of the negentropy of the
    softmax scores plus the squared difference of maximum softmax scores and minimum softmax
    scores.'
"""

###########################
# Token Level Measures
###########################

# 3 * 7 = 21
LMn_tok = None
LMe_tok = None
LSt_tok = None
LV_tok = None
LMi_tok = None
LMa_tok = None
LMc_tok = None
SMn_tok = None
SMe_tok = None
SSt_tok = None
SV_tok = None
SMi_tok = None
SMa_tok = None
SMc_tok = None
LnMn_tok = None
LnMe_tok = None
LnSt_tok = None
LnV_tok = None
LnMi_tok = None
LnMa_tok = None
LnMc_tok = None

SE_tok = None

# 3 * 7 = 21
LMns_tok = None
LMes_tok = None
LVs_tok = None
LMis_tok = None
LMas_tok = None
LMcs_tok = None
SMns_tok = None
SMes_tok = None
SVs_tok = None
SMis_tok = None
SMas_tok = None
SMcs_tok = None
LnMns_tok = None
LnMes_tok = None
LnVs_tok = None
LnMis_tok = None
LnMas_tok = None
LnMcs_tok = None

SEs_tok = None

# 1 p 5 p 3 = 9
LStsr_tok = None
SMnsr_tok = None
SMesr_tok = None
SStsr_tok = None
SMisr_tok = None
SMasr_tok = None
LnMnsr_tok = None
LnStsr_tok = None
LnMasr_tok = None

SEsr_tok = None

# 7
SdLnMn_tok = None
SdLnMe_tok = None
SdLnSt_tok = None
SdLnV_tok = None
SdLnMi_tok = None
SdLnMa_tok = None

# 7
SxLMn_tok = None
SxLMe_tok = None
SxLSt_tok = None
SxLV_tok = None
SxLMi_tok = None
SxLMa_tok = None
SxLMc_tok = None

# 7
SxLMn_tok = None
SxLMe_tok = None
SxLSt_tok = None
SxLV_tok = None
SxLMi_tok = None
SxLMa_tok = None
SxLMc_tok = None

# 6
SxLMns_tok = None
SxLMes_tok = None
SxLVs_tok = None
SxLMis_tok = None
SxLMas_tok = None
SxLMcs_tok = None

# 7
LdSxLMn_tok = None
LdSxLMe_tok = None
LdSxLSt_tok = None
LdSxLV_tok = None
LdSxLMi_tok = None
LdSxLMa_tok = None
LdSxLMc_tok = None

# 7
SqLnMn_tok = None
SqLnMe_tok = None
SqLnSt_tok = None
SqLnV_tok = None
SqLnMi_tok = None
SqLnMa_tok = None

# 7
LnqSMn_tok = None
LnqSMe_tok = None
LnqSSt_tok = None
LnqSV_tok = None
LnqSMi_tok = None
LnqSMa_tok = None

LMadMn_tok = None
SMadMn_tok = None
LnMadMn_tok = None

# 6
SdLndx_tok = None
SxLdx_tok = None
LdSxLdx_tok = None
LnqSdx_tok = None
Ldx_tok = None

VSxL_tok = None
VSdLn_tok = None

VSxLs_tok = None
VSdLns_tok = None

VSxLsr_tok = None
VSdLnsr_tok = None

MadMnL_tok = None
MadMnLs_tok = None
MadMnLsr_tok = None

MadMnS_tok = None
MadMnSs_tok = None
MadMnSsr_tok = None

MadMnLn_tok = None
MadMnLns_tok = None
MadMnLnsr_tok = None

logits_scores_tok = None
softmax_scores_tok = None
norm_logits_scores_tok = None

def initialize_tensors(batch_size, max_new_tokens, device, precision, total_seq_batches):
    ###########################
    # Token Level Measures
    ###########################

    # 3 x 7 = 21 measures
    global LMn_tok
    global LMe_tok
    global LSt_tok
    global LV_tok
    global LMi_tok
    global LMa_tok
    global LMc_tok
    global SMn_tok
    global SMe_tok
    global SSt_tok
    global SV_tok
    global SMi_tok
    global SMa_tok
    global SMc_tok
    global LnMn_tok
    global LnMe_tok
    global LnSt_tok
    global LnV_tok
    global LnMi_tok
    global LnMa_tok
    global LnMc_tok

    global SE_tok

    # 3 * 6 = 18
    global LMns_tok
    global LMes_tok
    global LVs_tok
    global LMis_tok
    global LMas_tok
    global LMcs_tok
    global SMns_tok
    global SMes_tok
    global SVs_tok
    global SMis_tok
    global SMas_tok
    global SMcs_tok
    global LnMns_tok
    global LnMes_tok
    global LnVs_tok
    global LnMis_tok
    global LnMas_tok
    global LnMcs_tok

    global SEs_tok

    # 1 p 5 p 3 = 9
    global LStsr_tok
    global SMnsr_tok
    global SMesr_tok
    global SStsr_tok
    global SMisr_tok
    global SMasr_tok
    global LnMnsr_tok
    global LnStsr_tok
    global LnMasr_tok

    global SEsr_tok

    # 7
    global SdLnMn_tok
    global SdLnMe_tok
    global SdLnSt_tok
    global SdLnV_tok
    global SdLnMi_tok
    global SdLnMa_tok

    # 7
    global SxLMn_tok
    global SxLMe_tok
    global SxLSt_tok
    global SxLV_tok
    global SxLMi_tok
    global SxLMa_tok
    global SxLMc_tok

    # 6
    global SxLMns_tok
    global SxLMes_tok
    global SxLVs_tok
    global SxLMis_tok
    global SxLMas_tok
    global SxLMcs_tok

    # 7
    global LdSxLMn_tok
    global LdSxLMe_tok
    global LdSxLSt_tok
    global LdSxLV_tok
    global LdSxLMi_tok
    global LdSxLMa_tok
    global LdSxLMc_tok

    # 7
    global SqLnMn_tok
    global SqLnMe_tok
    global SqLnSt_tok
    global SqLnV_tok
    global SqLnMi_tok
    global SqLnMa_tok

    # 7
    global LnqSMn_tok
    global LnqSMe_tok
    global LnqSSt_tok
    global LnqSV_tok
    global LnqSMi_tok
    global LnqSMa_tok

    global LMadMn_tok
    global SMadMn_tok
    global LnMadMn_tok

    # 6
    global SdLndx_tok
    global SxLdx_tok
    global LdSxLdx_tok
    global LnqSdx_tok
    global Ldx_tok

    global VSxL_tok
    global VSdLn_tok

    global VSxLs_tok
    global VSdLns_tok

    global VSxLsr_tok
    global VSdLnsr_tok

    global MadMnL_tok
    global MadMnLs_tok
    global MadMnLsr_tok

    global MadMnS_tok
    global MadMnSs_tok
    global MadMnSsr_tok

    global MadMnLn_tok
    global MadMnLns_tok
    global MadMnLnsr_tok

    global logits_scores_tok
    global softmax_scores_tok
    global norm_logits_scores_tok

    ###########################
    # Token Level Measures
    ###########################

    # 3 x 7 = 21 measures
    LMn_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LMe_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LSt_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LV_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LMi_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LMa_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LMc_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SMn_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SMe_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SSt_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SV_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SMi_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SMa_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    SMc_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnMn_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnMe_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnSt_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnV_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnMi_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnMa_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)
    LnMc_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)

    SE_tok = torch.full((batch_size, total_seq_batches, max_new_tokens), float('nan'), device=device).to(precision)

    # 3 x 6 = 18 measures
    LMns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LMes_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LVs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LMis_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LMas_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LMcs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMes_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SVs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMis_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMas_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMcs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMes_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnVs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMis_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMas_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMcs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    SEs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 1 p 5 p 3 = 9 measures
    LStsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMnsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMesr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SStsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMisr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMasr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMnsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnStsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMasr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    SEsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 7
    SdLnMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SdLnMe_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SdLnSt_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SdLnV_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SdLnMi_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SdLnMa_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 7
    SxLMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMe_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLSt_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLV_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMi_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMa_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMc_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 6
    SxLMns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMes_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLVs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMis_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMas_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLMcs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 7
    LdSxLMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLMe_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLSt_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLV_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLMi_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLMa_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLMc_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 7
    SqLnMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SqLnMe_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SqLnSt_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SqLnV_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SqLnMi_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SqLnMa_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 7
    LnqSMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSMe_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSSt_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSV_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSMi_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSMa_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    LMadMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SMadMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnMadMn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    # 6
    SdLndx_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    SxLdx_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LdSxLdx_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    LnqSdx_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    Ldx_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    VSxL_tok = torch.full((batch_size, total_seq_batches, 1), float('nan'), device=device).to(precision)
    VSdLn_tok = torch.full((batch_size, total_seq_batches, 1), float('nan'), device=device).to(precision)
    
    VSxLs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    VSdLns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    VSxLsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    VSdLnsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    MadMnL_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnLs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnLsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    MadMnS_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnSs_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnSsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    MadMnLn_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnLns_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)
    MadMnLnsr_tok = torch.full((batch_size, total_seq_batches), float('nan'), device=device).to(precision)

    logits_scores_tok = torch.full((batch_size, total_seq_batches, 1), float('nan'), device=device).to(precision)
    softmax_scores_tok = torch.full((batch_size, total_seq_batches, 1), float('nan'), device=device).to(precision)
    norm_logits_scores_tok = torch.full((batch_size, total_seq_batches, 1), float('nan'), device=device).to(precision)

