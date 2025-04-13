import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TextIteratorStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
import transformers
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval import tasks
from lm_eval import evaluator

import math
import logging
import json
import os
from threading import Thread

from tqdm import tqdm

# --- Configuration Section ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
EVAL_TASKS = ["hellaswag", "winogrande", "arc_easy"]
NUM_FEWSHOT = 0
BATCH_SIZE = 4
DEVICE = "cuda:0"
MAX_LENGTH = 2048
LIMIT = 100
OUTPUT_DIR = "./lm_eval_output_evaluate"

logging.basicConfig(level=logging.INFO)
logging.getLogger("lm_eval").setLevel(logging.INFO)

class ScaledForwardNoGradScale(torch.autograd.Function):
    """
    Scales the forward pass by a factor, unscales gradients in backward.
    """
    @staticmethod
    def forward(ctx, input_data, scale):
        ctx.scale = scale
        return input_data * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / ctx.scale, None

class MyLlamaForCausalLM(LlamaForCausalLM):
    """
    Custom LLaMA model enabling optional fused cross-entropy.
    """
    def __init__(self, config, use_fusedlce=False, **config_kwargs):
        super().__init__(config)
        self.use_fusedlce = use_fusedlce
        if self.use_fusedlce:
            try:
                from cut_cross_entropy import LinearCrossEntropy
                self.LCE = LinearCrossEntropy()
                logging.info("Using fused LinearCrossEntropy (cut_cross_entropy).")
            except ImportError:
                logging.error("Failed to import cut_cross_entropy. Disabling fused LCE.")
                self.use_fusedlce = False
                self.LCE = None
        else:
            self.LCE = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_ids=None,
        **kwargs,
    ):
        effective_labels = labels if labels is not None else label_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        loss = None

        if effective_labels is not None:
            logging.debug("Using fused LCE for loss.")
            scaled_weights = self.lm_head.weight
            fused_loss = self.LCE(
                hidden_states.to(torch.float16),
                scaled_weights.to(torch.float16),
                effective_labels.to(torch.long),
            )
            loss = fused_loss.to(torch.float32)
            print(f"  fused_loss.shape: {fused_loss.shape}")
            exit()

        logits = self.lm_head(hidden_states)

        if not return_dict:
            output_tuple = (logits,) + outputs[1:]
            return (loss,) + output_tuple if loss is not None else output_tuple

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

def load_model(config_kwargs):
    """
    Loads the base model, then copies its state dict into our custom MyLlamaForCausalLM.
    """
    base_model_id = BASE_MODEL_ID
    logging.info(f"Loading base model {base_model_id}...")
    source_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
    source_config = source_model.config
    logging.info("Base model loaded. Creating target model...")

    target_model = MyLlamaForCausalLM(config=source_config, use_fusedlce=True, **config_kwargs)
    logging.info("Copying weights...")
    target_model.load_state_dict(source_model.state_dict())
    logging.info("Weights copied.")
    del source_model
    return target_model.to(torch.bfloat16)

class CustomModelWrapper(LM):
    """
    LM Harness model wrapper for the custom LLaMA.
    """
    REQ_ARGS = {"tokenizer_name"}
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, model, tokenizer, batch_size=1, device="cpu", max_length=None):
        super().__init__()
        self._device = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
        logging.info(f"Using device '{self._device}'")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info(f"Set pad_token to eos_token (ID: {self.tokenizer.eos_token_id})")
        self._pad_token_id = self.tokenizer.pad_token_id
        self.model = model
        self.model.eval().to(self._device)
        self._batch_size = int(batch_size)
        resolved_max_len = getattr(self.model.config, "max_position_embeddings", None)

        if max_length is not None:
            resolved_max_len = int(max_length)
        elif resolved_max_len is None:
            resolved_max_len = getattr(self.tokenizer, "model_max_length", self._DEFAULT_MAX_LENGTH)
        if resolved_max_len is None:
            resolved_max_len = self._DEFAULT_MAX_LENGTH
            logging.warning(f"Max length defaulting to {resolved_max_len}")

        self._max_length = resolved_max_len
        logging.info(f"Batch size: {self._batch_size}, Max length: {self._max_length}")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = dict(kv.split("=", 1) for kv in arg_string.split(",") if kv)
        missing_args = cls.REQ_ARGS - set(args.keys())
        if missing_args:
            raise ValueError(f"Missing required args: {missing_args}")

        tokenizer_name = args["tokenizer_name"]
        batch_size = int(args.get("batch_size", 1))
        device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        max_length = args.get("max_length", None)
        if max_length: 
            max_length = int(max_length)

        logging.info(f"Loading tokenizer '{tokenizer_name}'")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        logging.info("Loading model")
        model = load_model({})

        logging.info("Instantiating CustomModelWrapper.")
        return cls(model=model, tokenizer=tokenizer, batch_size=batch_size, device=device, max_length=max_length)

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(self, string: str, add_special_tokens=True, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: torch.Tensor, skip_special_tokens=True, **kwargs):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs):
        inps = inps.to(self.device)
        attention_mask = (inps != self._pad_token_id).long().to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inps, attention_mask=attention_mask, return_dict=True)
        if not hasattr(outputs, "logits"):
            raise TypeError(f"Model output missing 'logits' attribute. Got: {type(outputs)}")
        return outputs.logits

    def loglikelihood(self, requests):
        results = []
        for i in tqdm(range(0, len(requests), self.batch_size), desc="loglikelihood"):
            batch_reqs = requests[i : i + self.batch_size]
            batch_inputs = []
            batch_context_lens_est = []
            batch_continuation_lens_est = []

            for req_idx, req_obj in enumerate(batch_reqs):
                context, continuation = req_obj.args
                context_enc = self.tokenizer.encode(context, add_special_tokens=False)
                continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
                full_enc = self.tokenizer.encode(context + continuation, add_special_tokens=True)

                if len(full_enc) > self.max_length:
                    full_enc = full_enc[len(full_enc) - self.max_length :]

                batch_inputs.append(full_enc)
                final_len = len(full_enc)
                est_cont_len = min(len(continuation_enc), final_len)
                est_ctx_len = final_len - est_cont_len
                batch_context_lens_est.append(est_ctx_len)
                batch_continuation_lens_est.append(est_cont_len)

            batch_max_len = max(len(enc) for enc in batch_inputs)
            padded_input_ids = []
            for enc in batch_inputs:
                padding_len = batch_max_len - len(enc)
                padded_ids = enc + [self._pad_token_id] * padding_len
                padded_input_ids.append(padded_ids)

            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
            logits = self._model_call(input_ids_tensor)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_tensor[..., 1:].contiguous().to(self.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self._pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.shape)

            for j in range(len(batch_reqs)):
                cont_len = batch_continuation_lens_est[j]
                ctx_len = batch_context_lens_est[j]
                start_idx = max(0, ctx_len - 1)
                end_idx = start_idx + cont_len
                if start_idx >= loss.shape[1] or cont_len == 0:
                    log_likelihood = 0.0
                else:
                    actual_end_idx = min(end_idx, loss.shape[1])
                    if start_idx >= actual_end_idx:
                        log_likelihood = 0.0
                    else:
                        continuation_loss_slice = loss[j, start_idx:actual_end_idx]
                        log_likelihood = -continuation_loss_slice.sum().item()
                is_greedy = False
                results.append((log_likelihood, is_greedy))

        return results

    def generate_until(self, requests):
        results = []
        for req_idx, req_obj in tqdm(enumerate(requests), desc="generate_until", total=len(requests)):
            context_str, gen_kwargs = req_obj.args
            if not isinstance(gen_kwargs, dict):
                gen_kwargs = {}
            max_new_tokens = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            stop_sequences = gen_kwargs.get("until", None)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = (temperature > 0.0) or gen_kwargs.get("do_sample", False)

            input_ids = self.tok_encode(context_str, add_special_tokens=True)
            max_ctx_len = self.max_length - max_new_tokens
            if len(input_ids) > max_ctx_len:
                input_ids = input_ids[len(input_ids) - max_ctx_len :]

            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            hf_gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": self.eot_token_id,
                "pad_token_id": self._pad_token_id,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "top_p": gen_kwargs.get("top_p", None),
                "top_k": gen_kwargs.get("top_k", None),
            }
            hf_gen_kwargs = {k: v for k, v in hf_gen_kwargs.items() if v is not None}
            gen_config = transformers.GenerationConfig(**hf_gen_kwargs)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs=input_tensor,
                    generation_config=gen_config,
                )
            output_ids = generated_ids[0, input_tensor.shape[1] :]
            generated_text = self.tok_decode(output_ids, skip_special_tokens=True)

            if stop_sequences:
                for stop_str in stop_sequences:
                    stop_idx = generated_text.find(stop_str)
                    if stop_idx != -1:
                        generated_text = generated_text[:stop_idx]
                        break
            results.append(generated_text.strip())
        return results

    def loglikelihood_rolling(self, requests):
        logging.warning("loglikelihood_rolling not implemented. Returning NaN.")
        return [math.nan] * len(requests)

if __name__ == "__main__":
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_args_list = [
        f"tokenizer_name={TOKENIZER_NAME}",
        f"batch_size={BATCH_SIZE}",
        f"device={DEVICE}",
    ]
    if MAX_LENGTH:
        model_args_list.append(f"max_length={MAX_LENGTH}")
    model_args_string = ",".join(model_args_list)

    print("-" * 80)
    print("Starting LM Evaluation Harness")
    print(f"Tasks: {EVAL_TASKS}")
    print("Model Wrapper: CustomModelWrapper")
    print(f"Model Args: {model_args_string}")
    print(f"Num Few-shot: {NUM_FEWSHOT}")
    print(f"Limit: {LIMIT}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("-" * 80)

    print("Instantiating CustomModelWrapper...")
    try:
        wrapper_instance = CustomModelWrapper.create_from_arg_string(model_args_string)
        print("CustomModelWrapper instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate CustomModelWrapper: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Loading tasks: {EVAL_TASKS}")
    try:
        task_dict = tasks.get_task_dict(EVAL_TASKS)
        print("Tasks loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load tasks: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("Running evaluation using evaluator.evaluate...")
    try:
        results = evaluator.evaluate(
            lm=wrapper_instance,
            task_dict=task_dict,
            limit=LIMIT,
            log_samples=bool(OUTPUT_DIR),
            write_out=bool(OUTPUT_DIR),
        )
        print("Evaluation completed.")
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("-" * 80)
    print("Evaluation Results Summary:")
    serializable_results = json.loads(json.dumps(results, default=str))
    print(json.dumps(serializable_results, indent=4))
    print("-" * 80)

    if OUTPUT_DIR:
        summary_path = os.path.join(OUTPUT_DIR, "summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results summary saved to: {summary_path}")
        except Exception as e:
            print(f"Error saving results summary: {e}")
        print("\nCheck output directory for detailed task outputs.")

    print("\nEvaluation finished.")
