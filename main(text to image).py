import torch
from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from diffusers.utils import make_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device).manual_seed(31)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    variant="fp16"
).to(device)

prompt = "Ernest Hemingway in watercolour, cold color palette, muted colors, detailed, 8k"
images = pipeline(
    prompt,
    generator=generator
).images

make_image_grid(images, rows=1, cols=1)
for image_id, image in enumerate(images):
    image.save(f"generated image {image_id}.png")
pipeline.to("cpu")
del pipeline
torch.cuda.empty_cache()