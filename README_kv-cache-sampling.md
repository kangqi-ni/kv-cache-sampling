## KV Cache Sampling Implementation
This readme explains the implementation of kv cache sampling in streaming llm.

### Dependency
Please refer to README.md in LEval and README.md in Baseline/streaming_llm.

### Streaming LLM
We added the streaming_llm folder in the Baseline folder. The pos_shift folder reimplements the forward function in the Attention module of LLMs to enable streaming features. 
The core code on kv cache manipulation is in the class StartRecentKVCache within kv_cache.py: The original streaming llm simply retains the first few tokens and the recent tokens. To adapt it for samplig, it gets the attention scores and uses a reservior sampling strategy to accept/reject kv cache for each new incoming token while keeping the entire kv cache length within a specified number. When using the kv cache for generation, it offloads a number of kv cache with the lowest attention scores from the cache similar to Heavy Hitter Oracle (H2O).

Modifications in Streaming LLM to enable sampling in kv_cache.py.
- reservior_sampling either accepts or rejects a new incoming token and then returns the new kv cache.
- delete offloads a number of kv cache with the lowest attention scores.

### LEval 
We integrated the streaming llm code into Baselines/vicuna-test.py as Baselines/vicuna-streaming-llm.py (original streaming llm) and Baselines/vicuna-streaming-llm-sampling.py (streaming llm with sampling)

Modifications in LEval to enable streaming llm
- compute_past_kv_cache builds a kv cache from a document since LEval is essentially QA on long documents
- greedy_generate, taken from streaming llm, uses the pre-built kv cache and a new query to generate new tokens (the answer to the query based on the document) in a greedy selection manner.

To run LEval with streaming llm with or without sampling:
```
# Run streaming llm on LEval (the max length here is an arbitrary large number so not to truncate documents)
CUDA_VISIBLE_DEVICES=0 python Baselines/vicuna-streaming-llm.py --metric exam_eval --max_length 100k
```

```
# Run streaming llm on LEval with attention sampling (the max length here is an arbitrary large number so not to truncate documents)
CUDA_VISIBLE_DEVICES=0 python Baselines/vicuna-streaming-llm-sampling.py --metric exam_eval --max_length 100k --sampling
```