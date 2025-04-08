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

from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer.pad_token_id = tokenizer.eos_token_id

class ScaledForwardNoGradScale(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		return input

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None

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

		outputs = self.model(
			input_ids=input_ids,
			**kwargs
		)
		hidden_states = outputs.last_hidden_state
		h = hidden_states

		if label_ids is not None:
			if not  self.use_fusedlce:
				raise ValueError("  Not implemented! use_fusedlce is False, but label_ids is not None")

			if self.softmax_temperature == 1.0:
				return self.LCE(h.to(torch.float16), self.lm_head.weight.to(torch.float16), label_ids).to(torch.float32)
			loss = self.LCE(h.to(torch.float16), self.lm_head.weight.to(torch.float16), label_ids)
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

		streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		generation_kwargs = dict(input_ids=input_ids, streamer=streamer, generation_config=generation_config, tokenizer=tokenizer,)

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

		streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		generation_kwargs = dict(input_ids=input_ids, streamer=streamer, generation_config=generation_config, tokenizer=tokenizer,)

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

	target_model.model.embed_tokens.weight.data = source_model.model.embed_tokens.weight.data.clone()

	for i in range(len(source_model.model.layers)):

		target_model.model.layers[i].self_attn.q_proj.weight.data = source_model.model.layers[i].self_attn.q_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.k_proj.weight.data = source_model.model.layers[i].self_attn.k_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.v_proj.weight.data = source_model.model.layers[i].self_attn.v_proj.weight.data.clone()
		target_model.model.layers[i].self_attn.o_proj.weight.data = source_model.model.layers[i].self_attn.o_proj.weight.data.clone()

		target_model.model.layers[i].mlp.gate_proj.weight.data = source_model.model.layers[i].mlp.gate_proj.weight.data.clone()
		target_model.model.layers[i].mlp.up_proj.weight.data = source_model.model.layers[i].mlp.up_proj.weight.data.clone()
		target_model.model.layers[i].mlp.down_proj.weight.data = source_model.model.layers[i].mlp.down_proj.weight.data.clone()

		target_model.model.layers[i].input_layernorm.weight.data = source_model.model.layers[i].input_layernorm.weight.data.clone()
		target_model.model.layers[i].post_attention_layernorm.weight.data = source_model.model.layers[i].post_attention_layernorm.weight.data.clone()

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

		pretty_print_history(history)

		if do_once:
			do_once = 0
			message = "Oh, hey! It's so great to see you! What's up?"
		else:
			pretty_print_history(history, current="user")
			message = input("")

		history.append({"role": "user", "content": message})

		pretty_print_history(history)
		decoded = model.chat(history)
		history.append({"role": "assistant", "content": decoded})

def main():

	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

	accelerator = accelerate.Accelerator()

	model = load_model()
	model = accelerator.prepare(model)

	start_yapping(model)

if __name__ == "__main__":
    start_yapping()