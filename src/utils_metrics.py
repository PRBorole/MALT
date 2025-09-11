import numpy as np
import scipy
import torch
from jive.AJIVE import AJIVE

def get_covariance_matrix(A):
    '''
    Returns normalized covariance matrix

    Args:
    A: Matrix for which covariance is to be calculated

    Returns:
    cov_mat: covariance matrix
    '''
    mean_representation = A.mean(axis=0)
    A_sub = A - mean_representation
    A_sub_norm = A_sub/np.linalg.norm(A_sub,ord=2, axis=1).reshape(-1,1)

    cov_mat = (1/A_sub_norm.shape[0])*np.dot(A_sub_norm.T,A_sub_norm)
    return cov_mat

def get_erank(A, device='cuda'):
    '''
    Returns erank

    Args:
    A: Matrix for which erank is to be calculated

    Returns:
    erank: eRank of the matrix
    '''
    cov_mat = get_covariance_matrix(A)
    if 'cuda'!=device:
        S = scipy.linalg.svd(cov_mat, compute_uv=False)
    else:
        with torch.no_grad():
            cov_mat = torch.tensor(cov_mat).to(device)
            S = torch.linalg.svdvals(cov_mat).detach().cpu().numpy()
            cov_mat = cov_mat.detach().cpu()
    Q = min(A.shape[0], A.shape[1])
    sum_Q = np.sum(S[:Q])
    H = np.nansum([-(s/sum_Q)*np.log(s/sum_Q) for s in S[:Q]])
    erank = np.exp(H)

    return erank

def attention_entropy(attn_map: np.ndarray, mode="avg") -> float | np.ndarray:
    """
    Compute entropy of attention distributions.

    Parameters
    ----------
    attn_map : np.ndarray
        2D attention map [num_queries, num_keys].
    mode : str
        "row" -> return entropy per query token.
        "avg" -> return average entropy across queries.

    Returns
    -------
    float or np.ndarray
    """
    safe_attn = np.clip(attn_map, 1e-12, 1.0)

    if mode == "row":
        return -np.sum(safe_attn * np.log(safe_attn), axis=-1)

    elif mode == "avg":
        row_entropies = -np.sum(safe_attn * np.log(safe_attn), axis=-1)
        return row_entropies.mean()

    else:
        raise ValueError("mode must be 'row' or 'avg'")


def get_ajive_erank(blocks, init_signal_ranks, device='cuda'):
    '''
    Returns erank of ajive joint and individual components

    Args:
    blocks: Dictionary of the matrices
    init_signal_ranks: Dictionary initial signals (i.e. eRanks) of the blocks
    device: str, device to run on

    Returns:
    erank: Dictionary eRank of the blocks - J + I for each block
    '''

    ajive = AJIVE(init_signal_ranks=init_signal_ranks, device=device, n_jobs=4, n_randdir_samples=4)
    ajive.fit(blocks=blocks)

    results_dict = ajive.results_dict()
    erank = {}
    for k in results_dict.keys():
        if 'individual' in results_dict[k].keys():
            p_list = results_dict[k]['individual']['svals']/results_dict[k]['individual']['svals'].sum() 
        else:
            p_list = results_dict[k]['svals']/results_dict[k]['svals'].sum()
        p_list = p_list.to_list()
        erank[k] = np.exp(-np.sum([p*np.log(p) for p in p_list]))

    return erank


def get_matrix_entropy(A, device='cuda'):
    '''
    Returns matrix-based entropy of matrix from: 
    Skean, Oscar, et al. "Layer by layer: Uncovering hidden representations in language models." 
    arXiv preprint arXiv:2502.02013 (2025).

    Args:
    A: Matrix for which entropy is to be calculated
    device: str, device to run on

    Returns:
    H: Entropy
    '''
    
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)

    N, D = A.shape

    K = torch.matmul(A.transpose(0, 1), A)
    
    # if N > D:
    #     K = torch.matmul(A.transpose(0, 1), A) # N x N
    # else:
    #     K = torch.matmul(A, A.transpose(0, 1)) # D x D

    # A = torch.clamp(A, min=0) # cut negative values (not an issue for attention)
    K = K.to(device)
    K = K.double() / torch.trace(K.double())
    
    ek, _ = torch.linalg.eigh(K)
    K = K.detach().cpu().numpy()
    ek = ek.detach().cpu()

    mk = torch.gt(ek, 0.0)
    mek = ek[mk]

    mek = mek/mek.sum()
    H = -1*torch.sum(mek*torch.log(mek))

    return H.item()
