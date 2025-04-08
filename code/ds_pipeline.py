# exit()

import accelerate
import numpy as np
import time
import cache
import threading
import os
import ask_questions
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import packer_batcher
import torch
import model_loader
from tqdm import tqdm
import parser
import gemini_api
from safetensors.torch import load_file as load_safetensors
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs

import traceback
import datetime
import safetensors
from safetensors.torch import save_file
_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",)


def tokenizer(x):
	return _tokenizer(x, return_tensors="np", add_special_tokens=False,)['input_ids'][0]

def chat_tokenizer(x):
	return _tokenizer.apply_chat_template(
		[
			{
				"role": "user",
				"content": x,
			},
		],
		add_generation_prompt=True,
	)




def tokenize(input_dict, keys_to_tokenize, tokenizer=tokenizer,):

	if not isinstance(keys_to_tokenize, list):
		keys_to_tokenize = [keys_to_tokenize]
		
	out_dict = {}
	for key in keys_to_tokenize:
		if key in input_dict:
			out_dict[f"{key}_ids"] = tokenizer(input_dict[key])

	return out_dict




def prep_ds(ds, accelerator, batcher):
	process_id = accelerator.process_index
	num_processes = accelerator.num_processes

	lengths = []
	batches = []
	for question_idx, (question) in enumerate(ds):
		if (question_idx % num_processes) != process_id:
			continue
			
		input_ids = chat_tokenizer(question['input'])
		label_ids = np.int64(tokenizer(question['label']).tolist() + [_tokenizer.eos_token_id])

		lengths.append(len(input_ids) + len(label_ids))

		status = batcher.add(input_ids, label_ids)
		popped = batcher.pop()
		batches.append(popped)

		# if status in ["ready", "full"]:
		# 	popped = batcher.pop()
		# 	batches.append(popped)
		# 	status = batcher.add(input_ids, label_ids)
	
	if (0 < batcher.get_sample_count()):
		popped = batcher.pop()
		batches.append(popped)

	
	stats = {
		"max_length": max(lengths),
		"min_length": min(lengths),
		"mean_length": np.mean(lengths),
		"std_length": np.std(lengths),
	}
	return batches, stats

def build_string_from_dict(x, sep=', '):
	ret_str = []
	for key, value in x.items():
		if isinstance(value, float):
			value = f"{value:.4f}"
		elif isinstance(value, int):
			value = f"{value}"
		ret_str.append(f"{key}: {value}")
	return sep.join(ret_str)


def load_checkpoint(model, config, ckpt_path, sequence_length=None):
	accelerator = config["accelerator"]

	model = accelerator.unwrap_model(model)
	loaded_dict = load_safetensors(os.path.join(ckpt_path, "model.safetensors"), )
	loaded_dict = {k.split("_orig_mod.",1)[-1]:v for k,v in loaded_dict.items()}
	model.load_state_dict(loaded_dict, strict=False)
	if sequence_length is not None:
		model = update_sequence_length(model, config, sequence_length)
	# 	model = model.reset_freq_cis(sequence_length, old_context_length=config['sequence_length'])

	msd = model.state_dict()


	keys = model.state_dict().keys()
	keys = sorted(list(keys))
	for keys_idx, (_keys) in enumerate(keys):
		accelerator.print(f"  keys[{keys_idx}]: {_keys}  ({msd[_keys].shape})")

	reset_head = False

	if reset_head:
		vocab_head_key = None
		for msd_idx, (key, value) in enumerate(msd.items()):
			if "vocab_head" in key:
				vocab_head_key = key
				break
		
		assert(vocab_head_key is not None)

		try:
			std_head = 1.0 / math.sqrt(model.state_dict()[vocab_head_key].shape[-1])
		except Exception as e:
			accelerator.print(f"###" * 60,)
			accelerator.print(f"  Failed to calculate std_head")
			if accelerator.is_local_main_process:
				traceback.print_exc()
			accelerator.print(f"###" * 60,)
			for msd_idx, (key, value) in enumerate(msd.items()):
				try:
					accelerator.print(f"  msd[{key}]: {value.shape}")
				except:
					accelerator.print(f"  msd[{key}]: {msd_idx}")
			print(f"###" * 60,)
			# exit()
		nn.init.trunc_normal_(
			model.state_dict()[vocab_head_key],
			mean=0.0,
			std=std_head,
			a=-2 * std_head,
			b=2 * std_head
		)


	model = accelerator.prepare(model)
	return model


def run_evals(config_name="4ee634dd905b5d4b"):
	pass

	import accelerate
	accelerator = accelerate.Accelerator(
		mixed_precision="bf16",
		step_scheduler_with_optimizer=False,
	)

	our_dir = os.path.dirname(os.path.abspath(__file__))
	
	config = {
		"sequence_length": 1024,
		"batch_size": 64,
		"minimum_sequence_length": 32,

		"lr": 1e-4,
		"weight_decay": 0.1,
		"adam_epsilon": 1e-8,

		"num_warmup_steps": 20,
		"softmax_temperature": 1.0,

		"steps_between_evals": 128,
	}
	config.update(parser.config_parser().__dict__)

	config['batch_size'] //= accelerator.num_processes

	batcher = packer_batcher.Batcher(config=config)
	questions = ask_questions.load_questions()

	# ds_train = questions['train_questions']
	# ds_valid = questions['valid_questions']
	ds_test = questions['test_questions']
	# (train_batches, train_stats) = prep_ds(questions['train_questions'], accelerator=accelerator, batcher=batcher,)
	# (valid_batches, valid_stats) = prep_ds(questions['valid_questions'], accelerator=accelerator, batcher=batcher,)
	(test_batches, test_stats) = prep_ds(questions['test_questions'], accelerator=accelerator, batcher=batcher,)

	# config['num_training_steps'] = (len(ds_train) * 1)


	# config_hash = cache.quick_key(config)[:16]
	config_hash = config_name
	log_dir = os.path.join(our_dir, "logs", config_hash)
	checkpoint_dir = os.path.join(log_dir, "checkpoints")
	log_path = os.path.join(log_dir, "log.txt")
	config_path = os.path.join(log_dir, "config.json")

	print(f"  log_path: {log_path}")

	if not os.path.exists(log_path):
		exit()

	model = model_loader.load_model(config)


		
	for name, param in model.named_parameters():
		# print min max avg
		print(f"  {name}: {param.shape} -> {param.min().item():.4f}, {param.max().item():.4f}, {param.mean().item():.4f}")
		break

	def checkpoint_dir_to_idx(x):
		return int(x.split("_")[-1])
	
	checkpoint_dirs = os.listdir(checkpoint_dir)
	checkpoint_dirs = sorted(checkpoint_dirs, key=checkpoint_dir_to_idx)
	# latest_checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_dirs[-1])

	loaded = load_safetensors(os.path.join(checkpoint_dir, checkpoint_dirs[-1], "model.safetensors"), )
	loaded_keys = loaded.keys()
	for loaded_keys_idx, (_loaded_keys) in enumerate(loaded_keys):
		print(f"  loaded_keys[{loaded_keys_idx}]: {_loaded_keys}")
	
	model_keys = model.state_dict().keys()
	for model_keys_idx, (_model_keys) in enumerate(model_keys):
		print(f"  model_keys[{model_keys_idx}]: {_model_keys}")
	loaded = {k.split("_orig_mod.",1)[-1]:v for k,v in loaded.items()}
	model.load_state_dict(loaded, strict=False)
	model.lm_head.weight.data = model.get_input_embeddings().weight.data



	# load the checkpoint
	# model = accelerator.prepare(model)
	# accelerator.load_state(checkpoint_dir)

	# print out model param stats
	for name, param in model.named_parameters():
		# print min max avg
		print(f"  {name}: {param.shape} -> {param.min().item():.4f}, {param.max().item():.4f}, {param.mean().item():.4f}")
		break

	print(f"  Done!")

	# base dir
	base_dir = os.path.dirname(os.path.abspath(__file__))
	up_one_dir = os.path.split(base_dir)[0]
	test_questions_location = os.path.join(up_one_dir, "data", "TestSet", "testset.json")
	with open(test_questions_location, "r") as file:
		test_questions = json.load(file)
	
	test_questions = test_questions[accelerator.process_index::accelerator.num_processes]
	# model = accelerator.prepare(model)
	model = model.to(accelerator.device)
	accuracy = []
	for tq in test_questions:
		input_string = tq['input']
		answer = tq['answer']
		key = cache.quick_key(dict(config_hash=config_hash, input_string=input_string))
		try:
			output = cache.quick_load(key)
		except:
			input_messages = [
				{
					"role": "user",
					"content": input_string,
				}
			]
			output = model.chat(input_messages)
			cache.quick_save(key, output)

		print(f"*" * 60,)
		print(f"  mean_acc: {np.mean(accuracy)},  answer: {answer},  output: {output}")
		print(f"*" * 60,)
		if output.strip().startswith(answer.strip()) or output.strip() in answer or answer.strip() in output.strip():
			accuracy.append(1)
		else:
			accuracy.append(0)
		

	acc = np.mean(accuracy) * len(accuracy)
	num_accs = torch.tensor(len(accuracy), device=accelerator.device)
	num_accs = accelerator.gather(num_accs).sum().item()
	len_qs = torch.tensor(len(test_questions), device=accelerator.device)
	len_qs = accelerator.gather(len_qs).sum().item()
	acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).sum().item() / num_accs
	# final_loss = accelerator.gather(final_loss).mean().item()

	accuracy_path = os.path.join(log_dir, "accuracy.txt")
	if accelerator.is_main_process:
		with open(accuracy_path, "w") as f:
			f.write("mean_accuracy, len_accuracy, len_questions, pid\n")
			f.write(f"{acc}, {num_accs}, {len_qs}, {os.getpid()}\n")
			# f.write()
		
		with open(accuracy_path, "r") as f:
			print(f.read())

	accelerator.wait_for_everyone()
	accelerator.print(f"*" * 60,)
	accelerator.print(f"  Finished evaluation, accuracy: {acc:.4f}, num_accs: {num_accs}, len_qs: {len_qs},  softmax_temperature: {config['softmax_temperature']},  batch_size: {config['batch_size']},  process_index: {accelerator.process_index}")
	accelerator.print(f"  Saved results to: {accuracy_path}")
	accelerator.print(f"*" * 60,)


def has_model_run_eval(model, accelerator, step, config_hash, log_dir):

	# base dir
	base_dir = os.path.dirname(os.path.abspath(__file__))
	up_one_dir = os.path.split(base_dir)[0]
	test_questions_location = os.path.join(up_one_dir, "data", "TestSet", "testset.json")
	with open(test_questions_location, "r") as file:
		test_questions = json.load(file)
	
	test_questions = test_questions[accelerator.process_index::accelerator.num_processes]
	# model = accelerator.prepare(model)
	# model = model.to(accelerator.device)
	accuracy = []
	for tq in test_questions:
		input_string = tq['input']
		answer = tq['answer']
		key = cache.quick_key(dict(config_hash=config_hash, input_string=input_string))
		try:
			output = cache.quick_load(key)
		except:
			input_messages = [
				{
					"role": "user",
					"content": input_string,
				}
			]
			try:
				output = model.chat(input_messages)
			except:
				output = accelerator.unwrap_model(model).chat(input_messages)
			cache.quick_save(key, output)


		if output.strip().startswith(answer.strip()) or output.strip() in answer or answer.strip() in output.strip():
			accuracy.append(1)
		else:
			accuracy.append(0)
		

	acc = np.mean(accuracy) * len(accuracy)
	num_accs = torch.tensor(len(accuracy), device=accelerator.device)
	num_accs = accelerator.gather(num_accs).sum().item()
	len_qs = torch.tensor(len(test_questions), device=accelerator.device)
	len_qs = accelerator.gather(len_qs).sum().item()
	acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).sum().item() / num_accs
	# final_loss = accelerator.gather(final_loss).mean().item()

	accuracy_path = os.path.join(log_dir, "accuracy.txt")
	if accelerator.is_main_process:
		if not os.path.exists(accuracy_path):
			with open(accuracy_path, "w") as f:
				f.write("mean_accuracy, len_accuracy, len_questions, pid, step\n")
		with open(accuracy_path, "a") as f:
			f.write(f"{acc}, {num_accs}, {len_qs}, {accelerator.process_index}, {step}\n")
		
		with open(accuracy_path, "r") as f:
			print(f.read())

	accelerator.wait_for_everyone()
	# accelerator.print(f"*" * 60,)
	# accelerator.print(f"  Finished evaluation, accuracy: {acc:.4f}, num_accs: {num_accs}, len_qs: {len_qs},  softmax_temperature: {config['softmax_temperature']},  batch_size: {config['batch_size']},  process_index: {accelerator.process_index}")
	# accelerator.print(f"  Saved results to: {accuracy_path}")
	# accelerator.print(f"*" * 60,)
	return acc

	




# if __name__ == "__main__":
def main():
	print(f'  welcome!')

	# if os.environ.get("USER", "unknown") == "sam":
	# 	run_evals()
	# 	exit()

	accelerator = accelerate.Accelerator(
		mixed_precision="bf16",
		step_scheduler_with_optimizer=False,
		device_placement=True,
		kwargs_handlers=[
			accelerate.InitProcessGroupKwargs(
				# Can help reduce desyncs at the start of training, even though we have a wait_for_everyone
				# Helpful in case one host is randomly substantially slower to compile than the others
				timeout=datetime.timedelta(seconds=3000) 
			),
			DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16),

		],

	)

	our_dir = os.path.dirname(os.path.abspath(__file__))
	
	config = {
		"sequence_length": 1024,
		"batch_size": 64,
		"minimum_sequence_length": 32,

		"lr": 1e-4,
		"weight_decay": 0.1,
		"adam_epsilon": 1e-8,

		"num_warmup_steps": 20,
		"softmax_temperature": 1.0,

		"steps_between_evals": 1024,
	}
	config.update(parser.config_parser().__dict__)
	if ((1e-3 - config['lr']) < 1e-6):
		print(f'  lr is too high. exiting.')
		exit()
	# if ((config['lr'] - 5e-5) < 1e-6):
	# 	print(f'  lr is too low. exiting.')
	# 	exit()
	if config['lr_schedule'] != 'fixed':
		print(f'  lr_schedule is not fixed. exiting.')
		exit()
	if config['batch_size' != 64]:
		print(f'  batch_size is not 64. exiting.')
		exit()

	softmax_temperatures = [0.2, 0.4, 0.6, 0.8, 0.9999, 1.0, 1.0001, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0,]
	# softmax_temperatures = [0.9999, 1.0, 1.0001,]
	softmax_temperatures = [0.20, 0.33, 0.5, 0.8, 0.9, 1.00, 1.11, 1.25, 1.43, 1.67, 2.0, 3.0]
	softmax_temperatures = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, ]

	if config['softmax_temperature'] not in softmax_temperatures:
		print(f"  softmax_temperature is not in {softmax_temperatures}. exiting.")
		exit()

	config['batch_size'] //= accelerator.num_processes

	batcher = packer_batcher.Batcher(config=config)
	questions = ask_questions.load_questions()

	ds_train = questions['train_questions']
	ds_valid = questions['valid_questions']
	ds_test = questions['test_questions']
	(train_batches, train_stats) = prep_ds(questions['train_questions'], accelerator=accelerator, batcher=batcher,)
	(valid_batches, valid_stats) = prep_ds(questions['valid_questions'], accelerator=accelerator, batcher=batcher,)
	(test_batches, test_stats) = prep_ds(questions['test_questions'], accelerator=accelerator, batcher=batcher,)

	config['num_training_steps'] = int((len(ds_train) // config['batch_size']) * 32)

	config_kosher = {k: v for k, v in config.items()}
	config_kosher['batch_size'] = config['batch_size'] * accelerator.num_processes
	del config_kosher['num_training_steps']
	config_hash = cache.quick_key(config_kosher)[:16]

	config_hash = cache.quick_key(config)[:16]


	log_dir = os.path.join(our_dir, "logs", config_hash)
	checkpoint_dir = os.path.join(log_dir, "checkpoints")
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(checkpoint_dir, exist_ok=True)
	log_path = os.path.join(log_dir, "log.txt")
	config_path = os.path.join(log_dir, "config.json")

	if os.path.exists(log_path):
		exit()


	accelerator.print(f"  len(train_batches): {len(train_batches):,}, samples_per_batch: {(len(ds_train) / len(train_batches)):.1f}")
	accelerator.print(f"  len(valid_batches): {len(valid_batches):,},  samples_per_batch: {len(ds_valid) / len(valid_batches):.1f}")
	accelerator.print(f"  len(test_batches): {len(test_batches):,},  samples_per_batch: {len(ds_test) / len(test_batches):.1f}")
	

	model = model_loader.load_model(config)
	optimizer, scheduler = model_loader.prepare_optimizer_and_scheduler(model, config)

	num_training_steps = config['num_training_steps']

	(model, optimizer, scheduler) = accelerator.prepare(model, optimizer, scheduler,)


	progress_bar = tqdm(range(num_training_steps), desc="Training", disable=(notaccelerator.is_main_process))

	def gen(x):
		while True:
			for x_idx, (batch) in enumerate(x):
				batch = {k: torch.tensor(v, dtype=torch.long).to(accelerator.device) for k, v in batch.items()}
				yield batch
	ds_train_gen = gen(train_batches)

	min_validation_loss = float("inf")
	min_validation_loss_step = 0
	min_validation_acc = 0.0
	min_validation_acc_step = 0


	def run_eval():
		accelerator.wait_for_everyone()
		model.eval()
		with accelerator.no_sync(model):
			losses = []
			for batch in tqdm(valid_batches, desc="Validation", disable=(not accelerator.is_main_process)):
				batch['softmax_temperature'] = 1.0
				batch = {k: torch.tensor(v, dtype=torch.long).to(accelerator.device) for k, v in batch.items()}
				loss = model(**batch)
				losses.append(loss.item())
			final_loss = np.mean(losses)

		final_loss = torch.tensor(final_loss, device=accelerator.device) * len(losses)
		num_losses = torch.tensor(len(losses), device=accelerator.device)

		final_loss = accelerator.gather(final_loss).mean().item()
		num_losses = accelerator.gather(num_losses).sum().item()
		final_loss /= num_losses
		model.train()
		accelerator.wait_for_everyone()
		return final_loss
	

	
	def write_log(name, msg):
		name, filetype = name.rsplit(".", 1)
		name = f"{name}_{accelerator.process_index}.{filetype}"
		def _write():
			with open(log_path, "a") as f:
				f.write(msg + "\n")
		threading.Thread(target=_write).start()

	if accelerator.is_main_process:
		open(log_path, "w").close()
		with open(config_path, "w") as f:
			json.dump(config, f, indent=4)

		# write train, valid, test stats
		write_log("train_stats.txt", build_string_from_dict(train_stats, sep='\n',))
		write_log("valid_stats.txt", build_string_from_dict(valid_stats, sep='\n',))
		write_log("test_stats.txt", build_string_from_dict(test_stats, sep='\n',))

	def save_checkpoint(step_idx):
		accelerator.wait_for_everyone()
		current_checkpoint_dir = os.path.join(checkpoint_dir, f"step_{step_idx}")
		accelerator.print(f"  Saving checkpoint at step_idx: {step_idx:,} -> \"{current_checkpoint_dir}\"")
		accelerator.save_state(current_checkpoint_dir)
		# _model = accelerator.unwrap_model(model)
		# save_dict = _model.state_dict()
		# # safe tensors
		# save_file(save_dict, os.path.join(current_checkpoint_dir, "model.safetensors"), )
		# del _model
		# model = accelerator.prepare(model)
		accelerator.wait_for_everyone()


	initial_validation_acc = None
	greatest_validation_acc = -1.0
	for step in range(num_training_steps):
		batch = next(ds_train_gen)

		with accelerator.autocast():
			optimizer.zero_grad()
			batch['softmax_temperature'] = config['softmax_temperature']
			loss = model(**batch)
			accelerator.backward(loss)
			if (0.0 != config.get("clip_grad_norm", 1.0)):
				accelerator.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()

		if (step % config['steps_between_evals'] == 0) or (step == num_training_steps - 1):
			eval_loss = run_eval()
			acc = has_model_run_eval(model=model, accelerator=accelerator, step=step, config_hash=config_hash, log_dir=log_dir)

			initial_validation_acc = (initial_validation_acc or acc)
			greatest_validation_acc = max(greatest_validation_acc, acc)


			if (min_validation_acc < acc):
				min_validation_acc = acc
				min_validation_acc_step = step
				save_checkpoint(step)

			elif eval_loss < min_validation_loss:
				min_validation_loss = eval_loss
				min_validation_loss_step = step
				save_checkpoint(step)

			if (num_training_steps // 3) <= step:
				if (initial_validation_acc == greatest_validation_acc):
					accelerator.print(f"  Exiting early, initial_validation_acc == greatest_validation_acc, even though we're on training step {int(step):,} of {int(num_training_steps):,}")
					exit()

			to_print = {
				"step": step,
				"loss": loss.item(),
				"eval_loss": eval_loss,
				"lr": optimizer.param_groups[0]["lr"],
				"progress": f"{step}/{num_training_steps}",
				"num_training_steps": num_training_steps,
				"batch_size": config['batch_size'],
				"softmax_temperature": config['softmax_temperature'],
				"min_validation_loss": min_validation_loss,
				"min_validation_loss_step": min_validation_loss_step,
				"process_index": accelerator.process_index,
				"clip_grad_norm": config.get("clip_grad_norm", 1.0),
			}
			msg = build_string_from_dict(to_print)
			accelerator.print(msg)
			write_log("validation_loss.txt", msg)
			accelerator.wait_for_everyone()

		progress_bar.update(1)
		progress_bar.set_postfix({"loss": loss.item(), "eval_loss": eval_loss, "lr": optimizer.param_groups[0]["lr"],})

	# eval_loss = run_eval()
	save_checkpoint(step)
	acc = has_model_run_eval(model, accelerator, step, config_hash, log_dir)





	exit()