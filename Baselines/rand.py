# import torch

# # Define sample dimensions for demonstration
# batch_size, num_heads, seq_length, feature_size = 1, 32, 5, 128

# # Initialize random tensors for attn_weights and value_states
# attn_weights = torch.randn(batch_size, num_heads, 1, seq_length)
# value_states = torch.randn(batch_size, num_heads, seq_length, feature_size)

# # Original loop-based approach
# loop_result = torch.stack([torch.matmul(attn_weights[:,:,:,i:i+1], value_states[:,:,i:i+1,:]) for i in range(attn_weights.size(-1))], dim=2).squeeze(3)

# print(loop_result.size())


# attn_weights.reshape(batch_size, num_heads, seq_length, 1, 1)
# value_states.reshape(batch_size, num_heads, seq_length, 1, feature_size)
# vector_result = torch.matmul(attn_weights, value_states).squeeze(3)

# # Show the shape of the loop_result for reference
# print(vector_result.size())

# print(torch.equal(loop_result, vector_result))



import torch

# Assuming x is your tensor with shape [32, 1, 32, 21]
x = torch.randn(2, 5)  # Example tensor
x = x ** 2

# Calculate the sum along the last dimension
sums = x.sum(dim=-1, keepdim=True)

# Normalize by dividing each element by the sum along the last dimension
normalized_x = x / sums

# Check the shape to confirm it remains unchanged
print(normalized_x.shape)  # Should print torch.Size([32, 1, 32, 21])

# Optionally, you can check if the normalization worked by summing the elements along the last dimension again
# It should be close to 1 for each vector
print(normalized_x.sum(dim=-1).size())

print(x)
print(normalized_x)
print(normalized_x.sum(dim=-1))