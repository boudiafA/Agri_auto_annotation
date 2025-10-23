# from transformers import pipeline, BitsAndBytesConfig
# import torch

# # Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

# # Initialize pipeline with quantization
# pipe = pipeline(
#     "image-text-to-text",
#     model="google/gemma-3-12b-it",
#     model_kwargs={
#         "quantization_config": quantization_config,
#         "device_map": "auto"
#     },
#     torch_dtype=torch.bfloat16
# )

# # Usage with system message (as shown in Gemma 3 docs)
# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "cat.jpg"},
#             {"type": "text", "text": "What do you see in this image?"}
#         ]
#     }
# ]

# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])


from transformers import pipeline, BitsAndBytesConfig
import torch

# -------- Setup: 4-bit quantization for better memory efficiency --------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-it",
    model_kwargs={
        "quantization_config": quantization_config,
        "device_map": "auto",
    },
    torch_dtype=torch.bfloat16,
)

# -------- Class metadata (provided) --------
CLASS_NAME = "abelmoschus_esculentus"
CLASS_DESCRIPTION = (
    "Abelmoschus esculentus (L.) Moench, commonly known as okra or lady’s finger, "
    "is a flowering plant in the mallow family Malvaceae, widely cultivated in tropical "
    "and subtropical regions for its edible green seed pods. It is an erect, annual herb "
    "reaching 1–2 m in height, with hairy stems and large, lobed leaves. The plant bears "
    "attractive, hibiscus-like yellow flowers with a red or purple center, which develop "
    "into elongated, ribbed pods harvested while tender. Okra thrives in warm, sunny climates "
    "with well-drained, fertile soil and moderate moisture. The pods are rich in fiber, vitamins, "
    "and minerals, and their mucilaginous texture makes them valuable in soups and stews across many cuisines."
)

# -------- Strict instruction block tailored to your TASKS/RULES --------
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
            {"type": "image", "url": "224586705.jpeg"},  # Replace with your actual image file/path
            {"type": "text", "text": USER_INSTRUCTIONS.strip()},
        ],
    },
]

# -------- Generate --------
output = pipe(text=messages, max_new_tokens=220)
print(output[0]["generated_text"][-1]["content"])
