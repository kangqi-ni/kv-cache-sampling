import torch
import random
random.seed(42)

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        use_sampling=False,
        use_sampling_v2=False
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.rejected = 0
        self.accepted = 0

        self.past_attentions = None
        self.past_values = None
        self.use_sampling = use_sampling
    
    def reservior_sampling(self, past_key_values, attentions):
        results = []
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        new_attentions = []
        for i in range(len(past_key_values)):
            k, v = past_key_values[i] 

            
            new_k = []
            new_v = []

            new_curr_attentions = []
            
            for j in range(k.size(1)):

                probs = attentions[i][0][j][0]
                
                r = random.random() 
                targeted_prob = probs[self.start_size]
                
                if r > targeted_prob:
                    self.rejected += 1
                    
                    new_k.append(
                        torch.cat(
                            [
                                k[:, j, 0: self.start_size, ...],
                                k[:, j, seq_len - self.recent_size: seq_len, ...],
                            ],
                            dim=self.k_seq_dim-1,
                        )
                    )
                    
                    new_v.append(
                        torch.cat(
                            [
                                v[:, j, 0: self.start_size, ...],
                                v[:, j, seq_len - self.recent_size: seq_len, ...],
                                ],
                            dim=self.v_seq_dim-1,
                        )
                    )


                    new_curr_attentions.append(
                        torch.cat(
                            [
                                probs[0: self.start_size],
                                probs[seq_len - self.recent_size: seq_len],
                            ],
                            dim=0,
                        )
                    )

                else:
                    self.accepted += 1

                    index = torch.argmin(probs[:self.start_size])
                    
                    new_k.append(
                        torch.cat(
                            [
                                k[:, j, 0: index, ...],
                                k[:, j, index+1: seq_len, ...],
                            ],
                            dim=self.k_seq_dim-1,
                        )
                    )
                    
                    new_v.append(
                        torch.cat(
                            [
                                v[:, j, 0: index, ...],
                                v[:, j, index+1: seq_len, ...],
                            ],
                            dim=self.v_seq_dim-1,
                        )
                    )
                    
                    new_curr_attentions.append(
                        torch.cat(
                            [
                                probs[0: index],
                                probs[index+1: seq_len],
                            ],
                            dim=0,
                        )
                    )

            new_attentions.append(new_curr_attentions)
            new_k = torch.cat(new_k, 0).unsqueeze(0)
            new_v = torch.cat(new_v, 0).unsqueeze(0)
            
            results.append([
                new_k,
                new_v,
            ])
            
        self.past_attentions = new_attentions
        return results
        
    def __call__(self, past_key_values, values=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if seq_len <= self.cache_size:
            return past_key_values

        # Use reservior sampling
        if self.use_sampling:
            return self.reservior_sampling(past_key_values, values)

        # Use streaming llm
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def delete(self, past_key_values, attentions, n):
        results = []
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        for i in range(len(past_key_values)):
            k, v = past_key_values[i] 
            # attentions[i] is 1 x 40 x 1 x 205
            # k is 1 x 40 x 205 x 128
            # v is 1 x 40 x 205 x 128

            new_k = []
            new_v = []
            
            for j in range(k.size(1)):

                probs = attentions[i][j]
                _, indices = torch.topk(probs, n, largest=False)
                mask = torch.ones(probs.size(), dtype=torch.bool)
                mask[indices] = False
                new_k_tensor = k[:, j, mask, ...]
                new_v_tensor = v[:, j, mask, ...]
            
                new_k.append(new_k_tensor)
                new_v.append(new_v_tensor)

            new_k = torch.cat(new_k, 0).unsqueeze(0)
            new_v = torch.cat(new_v, 0).unsqueeze(0)

            results.append([
                new_k,
                new_v,
            ])

        return results
    
    def evict_for_space(self, past_key_values, num_coming, values=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        if self.use_sampling:
            past_key_values = self.delete(past_key_values, values, num_coming)
            return past_key_values
            
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        
    # # This function is not used
    # def evict_range(self, past_key_values, start, end):
    #     if past_key_values is None:
    #         return None
    #     seq_len = past_key_values[0][0].size(self.k_seq_dim)
    #     assert start <= end and end <= seq_len
    #     return [
    #         [
    #             torch.cat(
    #                 [
    #                     self.k_slice(k, 0, start),
    #                     self.k_slice(k, end, seq_len),
    #                 ],
    #                 dim=self.k_seq_dim,
    #             ),
    #             torch.cat(
    #                 [
    #                     self.v_slice(v, 0, start),
    #                     self.v_slice(v, end, seq_len),
    #                 ],
    #                 dim=self.v_seq_dim,
    #             ),
    #         ]
    #         for k, v in past_key_values
    #     ]