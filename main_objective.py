# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:11:27 2025

@author: Rohan
"""

import torch
import torch.nn as nn

def belief_state_objective(all_f, all_b, text_head, x):
    """
    Computes the belief state objective for a sequence-based model.
    
    Parameters:
        all_f (Tensor): Forward state tensor of shape (batch_size, T, hidden_dim).
        all_b (Tensor): Backward state tensor of shape (batch_size, T, hidden_dim).
        text_head (nn.Module): The prediction head module.
        x (Tensor): Input tensor of token indices with shape (batch_size, T).

    Returns:
        loss (Tensor): Computed loss value.
    """
    
    bs, T = x.shape  # Batch size and sequence length

    # Get forward and backward states
    forward_state = all_f
    backward_state = all_b.flip(1)  # Reverse sequence for backward state

    # Generate time indices
    ft = torch.arange(T, dtype=torch.int32)  # Forward time indices
    bt = torch.arange(T, dtype=torch.int32)  # Backward time indices

    # Get all possible forward-backward time combinations
    combinations = torch.cartesian_prod(ft, bt)

    # Keep only pairs where backward index is at least 2 steps ahead of forward index
    combinations = combinations[(combinations[:, 1] - combinations[:, 0] >= 2)]

    # Filter valid forward-backward index pairs
    fb_pairs = combinations.clone()
    fb_pairs = fb_pairs[combinations[:, 1] < T]  # Ensure backward index stays within bounds

    # Extract forward and backward indices
    f_idxs = fb_pairs[:, 0]
    b_idxs = fb_pairs[:, 1]

    # Compute intermediate indices for single-label extraction
    nt_idxs = (combinations[:, 0] + 1)

    # Retrieve corresponding forward and backward states
    f = forward_state[:, f_idxs]  # Forward features
    b = backward_state[:, b_idxs]  # Backward features

    # Extract single token labels at next time step for both forward and backward states
    single_labels_f = x[:, nt_idxs].unsqueeze(2)  # Forward labels
    single_labels_b = x[:, b_idxs].unsqueeze(2)  # Backward labels
    single_labels = torch.cat((single_labels_f, single_labels_b), dim=2)  # Combine labels

    # Compute logits using the text prediction head
    logits = text_head(torch.cat([f, b], dim=2))

    # Reshape logits for loss computation
    fb_numpairs = fb_pairs.shape[0]
    logits = logits.reshape((bs, fb_numpairs, 2, -1))
    logits = logits.reshape((bs * fb_numpairs * 2, -1))

    # Reshape labels accordingly
    single_labels = single_labels.reshape((bs * fb_numpairs * 2))

    # Compute cross-entropy loss
    loss = nn.CrossEntropyLoss()(logits, single_labels)

    return loss

# Main execution block
if __name__ == '__main__':
    # Define batch size, sequence length, and hidden dimension
    batch_size = 8
    T = 12  # Sequence length
    m = 512  # Hidden dimension
    num_tokens = 100  # Number of unique tokens

    # Define forward and backward encoders (dummy models)
    enc_F = nn.Sequential(
        nn.Embedding(num_tokens, m),
        nn.Linear(m, m)
    )

    enc_B = nn.Sequential(
        nn.Embedding(num_tokens, m),
        nn.Linear(m, m)
    )

    # Define the text prediction head
    text_head = nn.Sequential(
        nn.Linear(m * 2, m),
        nn.LeakyReLU(),
        nn.Linear(m, num_tokens * 2)
    )

    # Generate random input tensor representing token sequences
    x = torch.randint(0, num_tokens, size=(batch_size, T))

    # Compute forward and backward encoded states
    f = enc_F(x)
    b = enc_B(x)

    # Detach tensors to prevent automatic gradient computation
    _f = f.detach()
    _b = b.detach()
    _f.requires_grad = True
    _b.requires_grad = True

    # Compute belief state objective loss
    loss = belief_state_objective(_f, _b, text_head, x)

    # Compute gradients for text head over all prefix/suffix pairs
    loss.backward()

    # Update encoders with one backward pass
    f.backward(_f.grad)
    b.backward(_b.grad)
    
    print(loss)

