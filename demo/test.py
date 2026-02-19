from transformers import AutoProcessor, AutoModelForCausalLM
import json

processor = AutoProcessor.from_pretrained("google/functiongemma-270m-it", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it", dtype="auto", device_map="auto")

with open('func.json','r') as rfl:
    fn_schema = json.loads(rfl.read())

with open('data2.txt', 'r') as rfl:
    msg = [x for x in rfl.readlines() if x.strip()]

results = []
for _m in msg:
    print("invoke from message:    ", _m) 
    message = [
        # ESSENTIAL SYSTEM PROMPT:
        # This line activates the model's function calling logic.
        {
            "role": "developer",
            "content": "You are a model that can do function calling with the following functions"
        },
        {
            "role": "user", 
            "content": _m  #"What's the temperature in London?"
        }
    ]
    
    inputs = processor.apply_chat_template(message, tools=fn_schema, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    
    out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=512,temperature=.6, use_cache = True,)
    output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    print(output)
    # <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
    results.append({'question': _m, 'result': output})
    
with open('results.json', 'w') as wfl:
    wfl.write(json.dumps(results))
