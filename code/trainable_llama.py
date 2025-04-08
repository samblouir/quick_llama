import torch
import torch.nn as nn

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
# from wrap_model import load_model
from model_arch import load_model
from functools import partial

def prepare_optimizer_and_scheduler(model, config):

	num_steps = config.get("num_training_steps", 200)
	num_warmup_steps = config.get("num_warmup_steps", int(num_steps * 0.05))

	optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=config.get('adam_epsilon', 1e-8))

	if config['lr_schedule'] == 'linear':
		scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)
	elif config['lr_schedule'] == 'cosine':
		scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)
	elif config['lr_schedule'] == 'fixed':
		scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

	return (optimizer, scheduler)


def prepare_optimizer(model, config):
	optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=config.get('adam_epsilon', 1e-8))
	return optimizer	

def prepare_scheduler(optimizer, config):
	num_steps = config.get("num_training_steps", 200)
	num_warmup_steps = config.get("num_warmup_steps", int(num_steps * 0.05))

	if config['lr_schedule'] == 'linear':
		scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)
	elif config['lr_schedule'] == 'cosine':
		scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)
	elif config['lr_schedule'] == 'cosine_with_min_lr':
		min_lr_rate = config.get('min_lr_rate', 0.1)
		scheduler = transformers.get_scheduler(
			"cosine_with_min_lr",
			optimizer=optimizer,
			num_warmup_steps=num_warmup_steps,
			num_training_steps=num_steps,
			scheduler_specific_kwargs=dict(min_lr_rate=min_lr_rate),
		)
	elif config['lr_schedule'] == 'fixed':
		scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

	return scheduler
		

def _train_step(
		batch,
		model,
		optimizer=None,
		scheduler=None,
		accelerator=None,
		config=None,
		return_loss=True,
		*args,
		**kwargs,
	):
	'''
		Train step for the model
	'''
	with accelerator.autocast():
		optimizer.zero_grad()
		loss = model(**batch)
		accelerator.backward(loss)
		accelerator.clip_grad_norm_(model.parameters(), max_norm=config["clip_grad_norm"])
		optimizer.step()
		scheduler.step()

		optimizer = accelerator.unwrap_model(optimizer)
		lr = optimizer.param_groups[0]["lr"]
		for param_group in optimizer.param_groups:
			param_group["weight_decay"] = (config['weight_decay'] * lr)
		optimizer = accelerator.prepare(optimizer)

	if return_loss:
		return loss.detach().cpu()
	
	return None



class TrainableLlama():
	'''
		Trainable Llama model
	'''

	def __init__(self, config, accelerator=None):
		super().__init__()
		self.config = config
		
		self.accelerator = (accelerator or config['accelerator'])
		self.model = self.load_model(config)
		self.optimizer = self.load_optimizer(config)
		self.scheduler = self.load_scheduler(config)

	def unwrapped_optimizer(self):
		return self.accelerator.unwrap_model(self._optimizer)

	@property
	def optimizer(self):
		return self._optimizer
	
	@optimizer.setter
	def optimizer(self, optimizer):
		self._optimizer = optimizer

	@property
	def scheduler(self):
		return self._scheduler
	
	@scheduler.setter
	def scheduler(self, scheduler):
		self._scheduler = scheduler
	
	@property
	def accelerator(self):
		return self._accelerator
	
	@accelerator.setter
	def accelerator(self, accelerator):
		self._accelerator = accelerator

	@property
	def model(self):
		return self._model
	
	@model.setter
	def model(self, model):
		self._model = model

	@property
	def config(self):
		return self._config
	
	@config.setter
	def config(self, config):
		self._config = config

	# missing items should be redirected to the model
	# for example, TrainableLlama().x should be redirected to self.model.x
	def __getattr__(self, name):
		if hasattr(self.model, name):
			return getattr(self.model, name)
		raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
	


	
	def load_model(self, config):
		# Load the model here
		cfg = (config or self.config)
		model = load_model(cfg)
		return self.accelerator.prepare(model)
	
	def load_optimizer(self, config):
		cfg = (config or self.config)
		optimizer = prepare_optimizer(self.model, cfg)
		return self.accelerator.prepare(optimizer)
	
	def load_scheduler(self, config):
		cfg = (config or self.config)
		scheduler = prepare_scheduler(self.optimizer, cfg)
		return self.accelerator.prepare(scheduler)
	
	# def forward(self, batch=None, *args, **kwargs):
	# 	if batch is None:
	# 		batch = {}
	# 	return self.model(*args, **batch, **kwargs)
	
	def __call__(self, *args, **kwargs):
		return self.train_step(*args, **kwargs)
	
	def train_step(self, *args, **kwargs):
		if self.model.training:
			return_loss = kwargs.pop('return_loss', True)
		else:
			return_loss = True
			

		return _train_step(
			*args,
			batch=kwargs,
			model=self.model,
			optimizer=self.optimizer,
			scheduler=self.scheduler,
			accelerator=self.accelerator,
			config=self.config,
			return_loss=return_loss,
		)


