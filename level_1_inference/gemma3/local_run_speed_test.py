from transformers import pipeline, BitsAndBytesConfig
import torch
import time
from tqdm import tqdm
import gc

# -------- Define quantization configurations --------
quantization_configs = {
    "4-bit": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    "8-bit": BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
}

# -------- Define models to test --------
model_ids = [
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it"
]

# -------- Class metadata (provided) --------
CLASS_NAME = "abelmoschus_esculentus"
CLASS_DESCRIPTION = (
    "Abelmoschus esculentus (L.) Moench, commonly known as okra or lady's finger, "
    "is a flowering plant in the mallow family Malvaceae, widely cultivated in tropical "
    "and subtropical regions for its edible green seed pods. It is an erect, annual herb "
    "reaching 1–2 m in height, with hairy stems and large, lobed leaves. The plant bears "
    "attractive, hibiscus-like yellow flowers with a red or purple center, which develop "
    "into elongated, ribbed pods harvested while tender. Okra thrives in warm, sunny climates "
    "with well-drained, fertile soil and moderate moisture. The pods are rich in fiber, vitamins, "
    "and minerals, and their mucilaginous texture makes them valuable in soups and stews across many cuisines."
)

# -------- Strict instruction block --------
SYSTEM_INSTRUCTIONS = """
You are generating metadata for a single image.
TASKS
Write a concise caption of EXACTLY 4 sentences summarizing key observations.
Cover these, ONLY if clearly visible:
- crop name and/or scientific name
- crop type
- growth stage
- ground cover
- image perspective (use one of: 'top-down', 'oblique', 'side', 'macro', 'unknown')
- plant density and health
- environmental conditions
- visible elements (e.g., plant, tree, flower, leaf, soil, stem, pod, sky, background_vegetation)
Create EXACTLY 7 tags.
Tags must include ONLY:
- the plant name (common or scientific)
- crop type
- visible elements that appear in the image (e.g., leaf, soil, stem, pod, flower, sky, etc.)
RULES
- If any attribute is not visible or cannot be determined, use the word 'unknown' rather than guessing.
- Use clear, neutral language. No speculation.
- Output MUST be STRICT JSON with double quotes, no trailing commas, no extra commentary.
- Keys must be exactly: "tags" then "caption".
- Tags must be lowercase single words or snake_case terms; no spaces inside a tag.
OUTPUT FORMAT (strict JSON on one line):
{"tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"], "caption": "A 4-sentence caption."}
Respond with STRICT JSON on a single line and nothing else.
"""

USER_INSTRUCTIONS = f"""
Image class: {CLASS_NAME}
Class description: {CLASS_DESCRIPTION}
Analyze the provided image and produce the required JSON.
"""

# -------- Build messages following Gemma 3 chat format --------
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS.strip()}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "224586705.jpeg"},  # Your image file
            {"type": "text", "text": USER_INSTRUCTIONS.strip()},
        ],
    },
]

# -------- Store all results --------
all_results = {}

# -------- Loop through all combinations --------
for model_id in model_ids:
    model_name = model_id.split("/")[-1]  # Extract model name (e.g., gemma-3-12b-it)
    
    for quant_name, quant_config in quantization_configs.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} with {quant_name} quantization")
        print(f"{'='*70}")
        
        try:
            # -------- Create pipeline with quantization --------
            print(f"Loading pipeline...")
            pipe = pipeline(
                "image-text-to-text",
                model=model_id,
                model_kwargs={
                    "quantization_config": quant_config,
                    "device_map": "auto",
                },
                torch_dtype=torch.bfloat16,
            )
            
            print("Pipeline loaded successfully!")
            print(f"\nRunning inference 10 times...")
            
            # Timing variables
            times = []
            results = []
            
            # -------- Run inference 10 times --------
            for i in tqdm(range(10), desc="Processing"):
                start_time = time.time()
                
                output = pipe(text=messages, max_new_tokens=220)
                generated_text = output[0]["generated_text"][-1]["content"]
                
                end_time = time.time()
                inference_time = end_time - start_time
                times.append(inference_time)
                results.append(generated_text)
                
                # Show result for first iteration
                if i == 0:
                    print(f"\nFirst result:\n{generated_text}")
            
            # -------- Calculate statistics --------
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            # Store results
            config_key = f"{model_name}_{quant_name}"
            all_results[config_key] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_time": std_time,
                "total_time": sum(times),
                "unique_results": len(set(results))
            }
            
            print(f"\n{'='*50}")
            print(f"TIMING RESULTS:")
            print(f"{'='*50}")
            print(f"Average inference time: {avg_time:.2f} seconds")
            print(f"Minimum inference time: {min_time:.2f} seconds") 
            print(f"Maximum inference time: {max_time:.2f} seconds")
            print(f"Standard deviation: {std_time:.2f} seconds")
            print(f"Total time for 10 runs: {sum(times):.2f} seconds")
            
            # Consistency check
            unique_results = len(set(results))
            print(f"\nConsistency check: {unique_results} unique result(s) out of 10 runs")
            if unique_results == 1:
                print("✅ All results are identical (deterministic)")
            else:
                print("⚠️ Results vary between runs")
            
            # -------- Clean up to free memory --------
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error testing {model_name} with {quant_name}: {str(e)}")
            all_results[f"{model_name}_{quant_name}"] = {"error": str(e)}
            continue

# -------- Print comparison summary --------
print(f"\n\n{'='*70}")
print(f"SUMMARY - ALL CONFIGURATIONS")
print(f"{'='*70}\n")

print(f"{'Configuration':<40} {'Avg Time (s)':<15} {'Min Time (s)':<15} {'Max Time (s)'}")
print(f"{'-'*70}")

for config_key, results in all_results.items():
    if "error" in results:
        print(f"{config_key:<40} ERROR: {results['error'][:30]}")
    else:
        print(f"{config_key:<40} {results['avg_time']:<15.2f} {results['min_time']:<15.2f} {results['max_time']:.2f}")

# Find fastest configuration
valid_results = {k: v for k, v in all_results.items() if "error" not in v}
if valid_results:
    fastest = min(valid_results.items(), key=lambda x: x[1]['avg_time'])
    print(f"\n⚡ Fastest configuration: {fastest[0]} ({fastest[1]['avg_time']:.2f}s average)")
    
    slowest = max(valid_results.items(), key=lambda x: x[1]['avg_time'])
    print(f"🐌 Slowest configuration: {slowest[0]} ({slowest[1]['avg_time']:.2f}s average)")
    
    speedup = slowest[1]['avg_time'] / fastest[1]['avg_time']
    print(f"📊 Speedup (slowest vs fastest): {speedup:.2f}x")