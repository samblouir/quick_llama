import accelerate
from cut_cross_entropy import LinearCrossEntropy
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
import os
import transformers
from transformers import TextIteratorStreamer

# import threading
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# set pad
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
tokenizer.pad_token_id = tokenizer.eos_token_id




class ScaledForwardNoGradScale(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, scale):
		# Store the scale factor for use in the backward pass
		ctx.scale = scale
		# Scale the input in the forward pass
		return input * scale

	@staticmethod
	def backward(ctx, grad_output):
		# Unscale the gradient in the backward pass
		return grad_output / ctx.scale, None





class CausalLlama(LlamaForCausalLM):
	def __init__(self, config, use_fusedlce=False, **config_kwargs):
		super().__init__(config)
		self.use_fusedlce = use_fusedlce
		if self.use_fusedlce:
			self.LCE = LinearCrossEntropy()
		self.softmax_temperature = 1 / float(config_kwargs.get("softmax_temperature", 1.0))

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		label_ids: torch.LongTensor = None,
		**kwargs
	):
		kwargs.pop('labels', None)

		# for batch_idx,(key,value) in enumerate(kwargs.items()):
		# 	accelerator.print(f"  batch[{key}].shape: {value.shape}")

		outputs = self.model(
			input_ids=input_ids,
			**kwargs
		)
		hidden_states = outputs.last_hidden_state
		h = hidden_states
		
		if label_ids is not None:
			if not  self.use_fusedlce:
				raise ValueError("  Not implemented! use_fusedlce is False, but label_ids is not None")
			# logits = self.lm_head(hidden_states)
			# if self.softmax_temperature != 1.0:
			# 	logits = logits * self.softmax_temperature
			# loss = torch.nn.functional.cross_entropy(logits.reshape((-1, logits.shape[-1])), label_ids.reshape((-1,)).to(torch.long), reduction="mean",)
			
			if self.softmax_temperature == 1.0:
				return self.LCE(h.to(torch.float16), self.lm_head.weight.to(torch.float16), label_ids).to(torch.float32)
			scaled_weights = ScaledForwardNoGradScale.apply(self.lm_head.weight, self.softmax_temperature)
			loss = self.LCE(h.to(torch.float16), scaled_weights.to(torch.float16), label_ids)
			return loss.to(torch.float32)
		else:
			logits = self.lm_head(hidden_states)
			return CausalLMOutputWithPast(
				loss=None,
				logits=logits,
				past_key_values=outputs.past_key_values,
				hidden_states=outputs.hidden_states,
				attentions=outputs.attentions,
			)
		
	


	def chat(self, input_str, deterministic=True, **kwargs):

		if isinstance(input_str, str):
			conversation = [
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": input_str},
			]
		else:
			conversation = input_str

		input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
		new_length = (len(input_ids) + 256)
		# align to 512
		new_length = (new_length // 512 + 1) * 512

		if deterministic:
			generation_kwargs = dict()
		else:
			generation_kwargs = dict(
				temperature=2.0,
				min_p=0.025,
				top_p=0.95,
				top_k=40,
				do_sample=True,
			)

		generation_config = transformers.generation.configuration_utils.GenerationConfig(
			max_length=new_length,
			stop_strings=["<|eot_id|>"],
			use_cache=False,
			tokenizer=tokenizer,
			repetition_penalty=1.1,
			pad_token_id=tokenizer.eos_token_id,
			**generation_kwargs,
		)
		
		input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
		iid_shape = input_ids.shape
		if len(iid_shape) == 1:
			input_ids = input_ids.unsqueeze(0)

		# Use TextIteratorStreamer to stream tokens
		streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		generation_kwargs = dict(input_ids=input_ids, streamer=streamer, generation_config=generation_config, tokenizer=tokenizer,)

		# Generate tokens and stream them
		thread = Thread(target=self.generate, kwargs=generation_kwargs)
		thread.start()

		rv = []
		header_removed = False
		pretty_print_history(conversation, current="assistant")

		for new_text in streamer:
			if not header_removed:
				new_text = new_text.rsplit('<|end_header_id|>\n', 1)[-1]
				header_removed = True

			print(new_text, end='', flush=True)
			rv.append(new_text)


		thread.join()
		ret_str = ''.join(rv)
		ret_str = ret_str.rsplit('<|end_header_id|>', 1)[-1].rsplit('<|eot_id|>', 1)[0].strip()
		return ret_str
		


	def chat_raw(self, input_str, deterministic=True, **kwargs,):
		input_ids = tokenizer.encode(input_str)
		new_length = (len(input_ids) + 256)
		# align to 512
		new_length = (new_length // 512 + 1) * 512

		if deterministic:
			generation_kwargs = dict()
		else:
			generation_kwargs = dict(
				temperature=2.0,
				min_p=0.025,
				top_p=0.95,
				top_k=40,
				do_sample=True,
			)

		generation_config = transformers.generation.configuration_utils.GenerationConfig(
			max_length=new_length,
			stop_strings=["<|eot_id|>"],
			use_cache=False,
			tokenizer=tokenizer,
			repetition_penalty=1.1,
			**generation_kwargs,
		)
		
		input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
		iid_shape = input_ids.shape
		if len(iid_shape) == 1:
			input_ids = input_ids.unsqueeze(0)

		# Use TextIteratorStreamer to stream tokens
		streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		generation_kwargs = dict(input_ids=input_ids, streamer=streamer, generation_config=generation_config, tokenizer=tokenizer,)

		# Generate tokens and stream them
		thread = Thread(target=self.generate, kwargs=generation_kwargs)
		thread.start()

		rv = []
		for new_text in streamer:
			print(new_text, end='', flush=True)
			rv.append(new_text)
		thread.join()
		return ''.join(rv)
		
		
	def chat_from_input_ids(self, input_ids, **kwargs):
		new_length = (len(input_ids)+256)
		# align to 512
		new_length = (new_length // 512 + 1) * 512
		import transformers
		generation_config = transformers.generation.configuration_utils.GenerationConfig(
			max_length=new_length,
			stop_strings=["<|eot_id|>"],
			use_cache=False,
			temperature=0.0,
		)
		generated_ids = self.generate(input_ids, generation_config=generation_config, tokenizer=tokenizer)
		decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		decoded = [x.split("assistant<|end_header_id|>",1)[-1].strip().rsplit("<|eot_id|>",1)[0] for x in decoded]
		decoded = decoded[0]
		return decoded


		
def load_model(config,):

	instruct = config.get('instruct', False)
	if instruct:
		source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
	else:
		source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

	_config = LlamaConfig(
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
			"rope_type": "llama3"
		},
		attention_bias=False,
	)

	target_model = CausalLlama(config=_config, use_fusedlce=True, **config)

	# Copy embeddings
	target_model.model.embed_tokens.weight.data = source_model.model.embed_tokens.weight.data.clone()

	# For each layer
	for i in range(len(source_model.model.layers)):
		# Copy attention weights
		target_model.model.layers[i].self_attn.q_proj.weight.data = source_model.model.layers[i].self_attn.q_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.k_proj.weight.data = source_model.model.layers[i].self_attn.k_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.v_proj.weight.data = source_model.model.layers[i].self_attn.v_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.o_proj.weight.data = source_model.model.layers[i].self_attn.o_proj.weight.data.clone()

		# Copy MLP weights
		target_model.model.layers[i].mlp.gate_proj.weight.data = source_model.model.layers[i].mlp.gate_proj.weight.data.clone()
		target_model.model.layers[i].mlp.up_proj.weight.data = source_model.model.layers[i].mlp.up_proj.weight.data.clone()
		target_model.model.layers[i].mlp.down_proj.weight.data = source_model.model.layers[i].mlp.down_proj.weight.data.clone()

		# Copy Layer Norms
		target_model.model.layers[i].input_layernorm.weight.data = source_model.model.layers[i].input_layernorm.weight.data.clone()
		target_model.model.layers[i].post_attention_layernorm.weight.data = source_model.model.layers[i].post_attention_layernorm.weight.data.clone()

	# Copy final norm and lm_head
	target_model.model.norm.weight.data = source_model.model.norm.weight.data.clone()
	target_model.lm_head.weight.data = source_model.lm_head.weight.data.clone()

	del source_model
	return target_model.to(torch.bfloat16)

def pretty_print_history(history, current=None):

	if current is not None:
		history = [*history, {"role": current, "content": ""}]

	os.system("clear")
	print(f"*" * 60,)
	for item in history:
		formatted_role = item['role'].capitalize().ljust(12)
		content = item['content']
		content_stripped = content.strip()
		print(f"\t{formatted_role}: {content_stripped}", end='')
		if len(content):
			print('\n', "*" * 60, sep='',)
	print(flush=True, end='')

def start_yapping(model=None):

	if model is None:
		model = load_model()
		accelerator = accelerate.Accelerator()
		model = accelerator.prepare(model)

	do_once = 1

	history = [
		{"role": "system", "content": "You are a helpful assistant"},
	]

	while True:

		# print(f"\n" * 3, end='',)
		# print(f"#" * 60,)

		pretty_print_history(history)

		if do_once:
			do_once = 0
			message = "Oh, hey! It's so great to see you! What's up?"
		else:
			pretty_print_history(history, current="user")
			message = input("")

		history.append({"role": "user", "content": message})

		# print(f"*" * 60,)
		# decoded = model.chat_raw(message)
		pretty_print_history(history)
		decoded = model.chat(history)
		history.append({"role": "assistant", "content": decoded})

		# print(f"{decoded}")
		# print(f"#" * 60,)




def main():
	# source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

	# generate text
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")



	accelerator = accelerate.Accelerator()

	model = load_model()
	model = accelerator.prepare(model)

	start_yapping(model)


if __name__ == "__main__":
    start_yapping()






