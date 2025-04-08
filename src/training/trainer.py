import signal
import sys
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from tqdm import tqdm
from transformers import AutoTokenizer
import accelerate
import ask_questions
import cache
import datetime
import json
import numpy as np
import os
import packer_batcher
import parser
import safetensors
import threading
import time
import torch
import traceback
import trainable_llama
from safetensors.torch import save_file
import random

_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",)

def tokenizer(x):
	return _tokenizer(str(x), return_tensors="np", add_special_tokens=False,)['input_ids'][0]

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

def format_multiple_choice(input, choices, label, *args, **kwargs,):
	out_str = [
		input,
	]

	alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	for idx, (choice) in enumerate(choices):
		letter = alphabet[idx]
		out_str.append(f"({letter}) {choice}")
	label_str = out_str[label+1]
	out_str = "\n".join(out_str)
	return dict(
		input=out_str,
		choices=choices,
		label=label_str,
	)

def prep_ds(ds, config, split=None,):
	batcher = packer_batcher.Batcher(config=config)
	accelerator = config['accelerator']
	process_id = accelerator.process_index
	num_processes = accelerator.num_processes

	lengths = []
	batches = []

	ds_cache_key = dict(

		sequence_length=config['sequence_length'],
		batch_size=config['batch_size'],
		dataset=config['dataset'],
		split=split,
		num_processes=num_processes,
		process_id=process_id,
	)
	ds_cache_key = cache.quick_key(ds_cache_key)
	try:

		batches, lengths = cache.quick_load(ds_cache_key)
	except:

		progress_bar = tqdm(total=int(len(ds)//num_processes), desc="Tokenizing", disable=(accelerator.is_main_process == False))

		for question_idx, (question) in enumerate(ds):

			if (question_idx % num_processes) != process_id:
				continue

			progress_bar.update(1)

			if config.get("dataset", None) in ['hermes']:
				conversation = question['conversations']
				messages = []
				for message in conversation:
					role = message['from']
					if role == "human":
						role = "user"
					elif role == "gpt":
						role = "assistant"
					messages.append({
						"role": role,
						"content": message['value'],
					})
				input_ids = np.int64(_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))

				lid = _tokenizer(messages[-1]['content'])['input_ids']

				label_ids = np.int64(lid + [_tokenizer.eos_token_id])

			else:
				input_ids = _tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
				label_ids = np.int64(_tokenizer(messages[-1]['content']).tolist() + [_tokenizer.eos_token_id])

			lengths.append(len(input_ids) + len(label_ids))

			status = batcher.add(input_ids, label_ids)

			if status in ["ready", "full"]:
				popped = batcher.pop()
				assert(isinstance(popped, dict))
				batches.append(popped)
				status = batcher.add(input_ids, label_ids)

		if (0 < batcher.get_sample_count()):
			popped = batcher.pop()
			assert(isinstance(popped, dict))
			batches.append(popped)

		while True:
			try:
				cache.quick_save(ds_cache_key, (batches, lengths))
				(batches, lengths) = cache.quick_load(ds_cache_key)
				break
			except Exception as e:
				print(f"  Failed to cache batches: {e}")

				time.sleep(15 + random.random() * 30)
				continue

	accelerator.wait_for_everyone()

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
		"dataset": "hermes",
		"sequence_length": 4096,
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

	config_hash = config_name
	log_dir = os.path.join(our_dir, "logs", config_hash)
	checkpoint_dir = os.path.join(log_dir, "checkpoints")
	log_path = os.path.join(log_dir, "log.txt")
	config_path = os.path.join(log_dir, "config.json")

	print(f"  log_path: {log_path}")

	if not os.path.exists(log_path):
		exit()

	model = trainable_llama.TrainableLlama(config)

	for name, param in model.named_parameters():

		print(f"  {name}: {param.shape} -> {param.min().item():.4f}, {param.max().item():.4f}, {param.mean().item():.4f}")
		break

	def checkpoint_dir_to_idx(x):
		return int(x.split("_")[-1])

	checkpoint_dirs = os.listdir(checkpoint_dir)
	checkpoint_dirs = sorted(checkpoint_dirs, key=checkpoint_dir_to_idx)

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

	for name, param in model.named_parameters():

		print(f"  {name}: {param.shape} -> {param.min().item():.4f}, {param.max().item():.4f}, {param.mean().item():.4f}")
		break

	print(f"  Done!")

	base_dir = os.path.dirname(os.path.abspath(__file__))
	up_one_dir = os.path.split(base_dir)[0]
	test_questions_location = os.path.join(up_one_dir, "data", "TestSet", "testset.json")
	with open(test_questions_location, "r") as file:
		test_questions = json.load(file)

	test_questions = test_questions[accelerator.process_index::accelerator.num_processes]

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

	accuracy_path = os.path.join(log_dir, "accuracy.txt")
	if accelerator.is_main_process:
		with open(accuracy_path, "w") as f:
			f.write("mean_accuracy, len_accuracy, len_questions, pid\n")
			f.write(f"{acc}, {num_accs}, {len_qs}, {os.getpid()}\n")

		with open(accuracy_path, "r") as f:
			print(f.read())

	accelerator.wait_for_everyone()
	accelerator.print(f"*" * 60,)
	accelerator.print(f"  Finished evaluation, accuracy: {acc:.4f}, num_accs: {num_accs}, len_qs: {len_qs},  softmax_temperature: {config['softmax_temperature']},  batch_size: {config['batch_size']},  process_index: {accelerator.process_index}")
	accelerator.print(f"  Saved results to: {accuracy_path}")
	accelerator.print(f"*" * 60,)

def has_model_run_eval(model, accelerator, step, config_hash, log_dir):

	base_dir = os.path.dirname(os.path.abspath(__file__))
	up_one_dir = os.path.split(base_dir)[0]
	test_questions_location = os.path.join(up_one_dir, "data", "TestSet", "testset.json")
	with open(test_questions_location, "r") as file:
		test_questions = json.load(file)

	test_questions = test_questions[accelerator.process_index::accelerator.num_processes]

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

	return acc

def batch_ds(ds_split, config, **kwargs,):
	ds = ds_split

	return prep_ds(ds, config=config, **kwargs,)

def clean_dict_serializable(d):
	out_dict = {}
	for key, value in d.items():
		try:
			json.dumps(value)
			out_dict[key] = value
		except:
			d[key] = f"FAILED TO SERIALIZE: {str(value)}"
	return out_dict

import psutil
import os
import signal
import sys

def find_and_kill_accelerate_processes():
    """
    Finds all processes running a command containing 'accelerate',
    prints their PIDs and command lines, and attempts to terminate them.
    """
    target_processes = []

    print("Scanning for processes involving 'accelerate'...")

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pinfo = proc.info
            cmdline = pinfo['cmdline'] 

            if cmdline and any('accelerate' in arg.lower() for arg in cmdline):
                pid = pinfo['pid']

                if pid == os.getpid():
                    continue

                command_str = ' '.join(cmdline)
                print(f"  Found 'accelerate' process: PID={pid}, Command='{command_str}'")
                target_processes.append(proc) 

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):

            pass
        except Exception as e:

            print(f"  Error accessing info for a process: {e}", file=sys.stderr)

    if not target_processes:
        print("No running 'accelerate' processes found.")
        return

    print(f"\nAttempting to terminate {len(target_processes)} process(es)...")

    for proc in target_processes:
        try:
            pid_to_kill = proc.pid
            print(f"  Terminating PID {pid_to_kill} ({' '.join(proc.cmdline())})...")

            proc.terminate()
            print(f"    Sent SIGTERM to PID {pid_to_kill}.")

        except psutil.NoSuchProcess:

            print(f"  Process with PID {pid_to_kill} no longer exists.")
        except psutil.AccessDenied:
            print(f"  Permission denied to terminate PID {pid_to_kill}. Try running the script with administrator/sudo privileges.", file=sys.stderr)
        except Exception as e:

            print(f"  An error occurred while trying to terminate PID {pid_to_kill}: {e}", file=sys.stderr)

def main():
	print(f'  welcome!')

	accelerator = accelerate.Accelerator(
		mixed_precision="bf16",
		step_scheduler_with_optimizer=False,

		device_placement=True,
		kwargs_handlers=[
			accelerate.InitProcessGroupKwargs(

				timeout=datetime.timedelta(seconds=3000) 
			),
			DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16),
		],
	)

	our_dir = os.path.dirname(os.path.abspath(__file__))

	config = {

		"dataset": "hermes",
		"sequence_length": 4096,
		"batch_size": 32,
		"minimum_sequence_length": 64,

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

	softmax_temperatures = [0.2, 0.4, 0.6, 0.8, 0.9999, 1.0, 1.0001, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0,]

	softmax_temperatures = [0.20, 0.33, 0.5, 0.8, 0.9, 1.00, 1.11, 1.25, 1.43, 1.67, 2.0, 3.0, 5.0]

	if config['softmax_temperature'] not in softmax_temperatures:
		print(f"  softmax_temperature is not in {softmax_temperatures}. exiting.")
		exit()

	config['batch_size'] //= accelerator.num_processes
	config['accelerator'] = accelerator

	if config['dataset'] == "hermes":
		ds = load_dataset("teknium/OpenHermes-2.5")['train']

		ds = ds.train_test_split(test_size=0.01, seed=42)
		ds = {
			"train": ds['train'],
			"test": ds['test'],
			"validation": ds['test'],
		}

	ds_train = ds["train"]
	ds_valid = ds["validation"]
	ds_test = ds["test"]

	(train_batches, train_stats) = batch_ds(ds_train, config, split='train',)
	(valid_batches, valid_stats) = batch_ds(ds_valid, config, split="validation",)
	(test_batches, test_stats) = batch_ds(ds_test, config, split="test",)

	for x in train_batches:
		accelerator.print(f"*" * 60,)
		for x_idx,(key,value) in enumerate(x.items()):
			accelerator.print(f"  x[{key}].shape: {value.shape}")

		accelerator.print(f"*" * 60,)
		assert(isinstance(x, dict))
		break

	num_training_steps = int((len(train_batches) * config['num_epochs']) - 2)
	config['num_training_steps'] = int(num_training_steps)
	config['num_warmup_steps'] = int(config['num_training_steps'] * 0.05)

	import copy
	config_kosher = {k: v for k, v in config.items() if k not in ['accelerator']}
	config_kosher['batch_size'] = config['batch_size'] * accelerator.num_processes

	config_hash = cache.quick_key(config_kosher)[:16]

	log_dir = os.path.join(our_dir, "logs", config_hash)
	if os.path.exists(log_dir):
		accelerator.print(f"  log_dir already exists. Exiting! log_dir: {log_dir}")
		exit()

	checkpoint_dir = os.path.join(log_dir, "checkpoints")
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(checkpoint_dir, exist_ok=True)
	log_path = os.path.join(log_dir, "log.txt")
	config_path = os.path.join(log_dir, "config.json")

	if os.path.exists(log_path):
		accelerator.print(f"  log_path already exists. Exiting! log_path: {log_path}")
		exit()

	accelerator.print(f"  len(train_batches): {len(train_batches):,}, samples_per_batch: {(len(ds_train) / len(train_batches)):.1f}")
	accelerator.print(f"  len(valid_batches): {len(valid_batches):,},  samples_per_batch: {len(ds_valid) / len(valid_batches):.1f}")
	accelerator.print(f"  len(test_batches): {len(test_batches):,},  samples_per_batch: {len(ds_test) / len(test_batches):.1f}")

	model = trainable_llama.TrainableLlama(config)

	num_training_steps = config['num_training_steps']

	progress_bar = tqdm(range(num_training_steps), desc="Training", disable=(accelerator.is_main_process == False))

	def gen(x):
		while True:
			for x_idx, (batch) in enumerate(x):
				if not len(batch):
					continue
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
			accelerator.wait_for_everyone()
			losses = []
			for batch in tqdm(valid_batches, desc="Validation", disable=(accelerator.is_main_process == False)):
				if not len(batch):
					continue
				batch['softmax_temperature'] = 1.0
				batch = {k: torch.tensor(v, dtype=torch.long).to(accelerator.device) for k, v in batch.items()}
				loss = model(**batch, return_loss=True)
				losses.append(loss.item())
			final_loss = np.mean(losses)

		accelerator.wait_for_everyone()

		final_loss = torch.tensor(final_loss, device=accelerator.device) * len(losses)
		num_losses = torch.tensor(len(losses), device=accelerator.device)

		final_loss = accelerator.gather(final_loss).mean().item()

		accelerator.wait_for_everyone()
		num_losses = accelerator.gather(num_losses).sum().item()

		accelerator.wait_for_everyone()
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
			json.dump(clean_dict_serializable(config), f, indent=4)

		write_log("train_stats.txt", build_string_from_dict(train_stats, sep='\n',))
		write_log("valid_stats.txt", build_string_from_dict(valid_stats, sep='\n',))
		write_log("test_stats.txt", build_string_from_dict(test_stats, sep='\n',))

	def save_checkpoint(step_idx):
		accelerator.print(f"  save_checkpoint({step_idx}):  Waiting for everyone...")
		accelerator.wait_for_everyone()
		accelerator.print(f"  save_checkpoint({step_idx}):  Saving!")
		current_checkpoint_dir = os.path.join(checkpoint_dir, f"step_{step_idx}")
		accelerator.print(f"  save_checkpoint({step_idx}):  Saving to: {current_checkpoint_dir}")
		accelerator.save_state(current_checkpoint_dir)
		accelerator.print(f"  save_checkpoint({step_idx}):  Saved!")

		accelerator.wait_for_everyone()
		accelerator.print(f"  save_checkpoint({step_idx}):  Waiting for everyone before returning...")

	if accelerator.is_main_process:
		to_write = []
		for config_idx,(key,value) in enumerate(config.items()):
			to_write.append(f"{key}: {value}")
		to_write = "\n".join(to_write)
		write_log("config.txt", to_write)
		write_log("config_hash.txt", config_hash)

	accelerator.wait_for_everyone()

	initial_validation_acc = None
	greatest_validation_acc = -1.0
	initial_validation_loss = None
	config_softmax_temperature = torch.tensor([config['softmax_temperature']]).to(accelerator.device).to(torch.float16)

	for step in range(num_training_steps):
		accelerator.wait_for_everyone()
		batch = next(ds_train_gen)
		batch['softmax_temperature'] = config_softmax_temperature
		accelerator.wait_for_everyone()
		loss = model(**batch,)

		if (step % config['steps_between_evals'] == 0) or (step == num_training_steps - 1):
			accelerator.wait_for_everyone()
			eval_loss = run_eval()
			accelerator.wait_for_everyone()

			initial_validation_loss = (initial_validation_loss or eval_loss)
			acc = (initial_validation_loss - eval_loss) / initial_validation_loss

			initial_validation_acc = (initial_validation_acc or acc)
			greatest_validation_acc = max(greatest_validation_acc, acc)

			save_checkpoint(step)

			to_print = {
				"step": step,
				"loss": loss.item(),
				"eval_loss": eval_loss,
				"lr": model.unwrapped_optimizer().param_groups[0]["lr"],
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

		if ((num_training_steps - 3) < step):
			print(f"  ATTEMPTING TO QUIT")
			break

		progress_bar.update(1)
		progress_bar.set_postfix({"loss": loss.item(), "eval_loss": eval_loss, "lr": model.unwrapped_optimizer().param_groups[0]["lr"],})

	print(f'  Done!', flush=True,)
	find_and_kill_accelerate_processes()

