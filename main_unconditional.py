import torch
from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from diffusers.utils import make_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device).manual_seed(31)

model_id = "google/ddpm-celebahq-256"
pipeline = DiffusionPipeline.from_pretrained(model_id).to(device)
images = pipeline(generator=generator).images
make_image_grid(images, rows=1, cols=1)
for image_id, image in enumerate(images):
    image.save(f"generated image {image_id}.png")