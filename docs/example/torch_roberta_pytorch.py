from transformers import AutoTokenizer, RobertaForQuestionAnswering
import torch
import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_sendnn import torch_sendnn

from torch.profiler import profile, record_function, ProfilerActivity

torch.utils.rename_privateuse1_backend("aiu")
torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
torch.utils.generate_methods_for_privateuse1_backend()

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

question, text = "Who was Miss Piggy?", "Miss Piggy was a muppet"

inputs = tokenizer(question, text, return_tensors="pt", max_length=384, padding="max_length")
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], 
                 record_shapes=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/roberta')) as prof:
        model = torch.compile(model, backend="sendnn")
        outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        print("-"*50)
        print('Answer: "{}"'.format(answer))
        print("="*50)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10).replace("CUDA", "AIU"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).replace("CUDA", "AIU"))