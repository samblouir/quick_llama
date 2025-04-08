from quick_llama.src.models.model_arch import (
	BaseModel,
	load_model,
	trainable_llama,
)
from transformers import AutoModelForCausalLM

config = dict(
	vocab_size=128256,
	hidden_size=2048,
	intermediate_size=8192,
	num_hidden_layers=16,
	num_attention_heads=32,
	num_key_value_heads=8,
	hidden_act="silu",
	max_position_embeddings=131072,
	initializer_range=0.02,
	rms_norm_eps=1e-5,
	use_cache=True,
	pad_token_id=None,
	bos_token_id=128000,
	eos_token_id=128001,
	pretraining_tp=1,
	tie_word_embeddings=True,
	rope_theta=500000.0,
	rope_scaling={
		"factor": 32.0,
		"high_freq_factor": 4.0,
		"low_freq_factor": 1.0,
		"original_max_position_embeddings": 8192,
		"rope_type": "llama3",
	},
	attention_bias=False,
)

model = BaseModel(config)
source_model = AutoModelForCausalLM.from_pretrained(
	"meta-llama/Llama-3.2-1B-Instruct"
).model

model = load_model(model, config)

for i in range(len(model.layers)):
	print(f"Layer {i}:")
	print(f"  {model.layers[i].__class__.__name__}")
	param_shapes = []
	for name, param in model.layers[i].named_parameters():
		if param.requires_grad:
			param_shapes.append(param.shape)
			# print std and mean
			print(
				f"  {name}: {param.shape}, std: {param.std()}, mean: {param.mean()}"
			)
