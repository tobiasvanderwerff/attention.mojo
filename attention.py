import numpy as np

def attention(Q, K, V):
    """ 
    Calculate the attention weights.

    Args:
        Q: query matrix, shape (batch_size, seq_len, d_model)
        K: key matrix, shape (batch_size, seq_len, d_model)
        V: value matrix, shape (batch_size, seq_len, d_model)

    Returns:
        weights: attention weights, shape (batch_size, seq_len, seq_len) 
    """
    QK = Q @ K.T
    dk = np.sqrt(K.shape[-1])
    QK = QK / dk
    weights = np.exp(QK)
    return weights


if __name__ == "__main__":
    Q = np.ones((2, 3))
    K = V = Q
    weights = attention(Q, K, V)
    print(weights)