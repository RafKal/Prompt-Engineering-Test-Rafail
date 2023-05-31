from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from PIL import Image
import uuid
import json
import os
import cleanfid

class PromptEngineeringBase:
    def __init__(self, pipe:DiffusionPipeline, scheduler, gpu:bool=True):
        self.pipe = pipe
        if scheduler == "DDIMScheduler":
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "LMDS":
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "Euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "Euler_A":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)  

        base_path = 'prompt_engineering_out_v'
        self.version = max(list(
            map(lambda x:int(x.replace(base_path,'')), 
                filter(lambda x:x.startswith(base_path), os.listdir("."))
            )
        )+[0])+1
        self.save_path = f"{base_path}{self.version}"
        
        os.mkdir(self.save_path)
        
        with open(f"{self.save_path}/annotations.txt","w") as json_file:
            pass

        if gpu:
            self.pipe = self.pipe.to("cuda")

        if scheduler == "DDIMScheduler":
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "LMDS":
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "Euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "Euler_A":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif scheduler == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def save_image(self, image:Image.Image, annotation:dict, prompt:str):
        image_uuid = str(uuid.uuid4())
        annotation["image_uuid"] = image_uuid
        with open(f"{self.save_path}/annotations.txt","a") as anno_txt:
            anno_txt.write(json.dumps(annotation))
        image.save(f"{self.save_path}/image_{prompt}.jpg")

        #print(f"{self.save_path}")

class PromptEngineeringInpainting(PromptEngineeringBase):
    def __init__(self, gpu:bool=True, scheduler:str=None):
        super().__init__(
            pipe=StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16),
                scheduler=scheduler,
                gpu=gpu
            )
        self.pipe:StableDiffusionInpaintPipeline = self.pipe
    def run(self, image:Image.Image, mask:Image.Image, prompt:str, negative_prompt:str, guidance_scale:float, images_per_prompt:int, num_inference_steps:int):
        result = self.pipe.__call__(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=images_per_prompt,
            height = image.height,
            width = image.width,
            num_inference_steps = num_inference_steps
        )
        out_images = result.images
        for out_image in out_images:
            self.save_image(out_image, {
                "prompt":prompt,
                "negative_prompt":negative_prompt,
                "guidance_scale":guidance_scale,

            },  prompt = prompt)

class PromptEngineeringText2Img(PromptEngineeringBase):
    def __init__(self, gpu:bool=True,scheduler:str=None):
        super().__init__(
            pipe=StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2", 
                torch_dtype=torch.float16
            ),
            scheduler=scheduler,
            gpu=gpu
        )
        self.pipe:StableDiffusionPipeline = self.pipe
    def run(self, prompt:str, negative_prompt:str, guidance_scale:float, height:int, width:int, images_per_prompt:int, num_inference_steps:int):
        result = self.pipe.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=images_per_prompt,
            num_inference_steps = num_inference_steps,
            height=height,
            width=width
        )
        out_images = result.images
        for out_image in out_images:
            self.save_image(out_image, {
                "prompt":prompt,
                "negative_prompt":negative_prompt,
                "guidance_scale":guidance_scale
            })
if __name__ == "__main__":
    inpaint_pipeline = PromptEngineeringInpainting(gpu=True,scheduler="DPM")
    text2img_pipeline = PromptEngineeringText2Img(gpu=True,scheduler="DPM")
    image = Image.open("./df6667e9-7573-41c4-b263-3fc4d4b88c93_image.jpg")
    mask = Image.open("./df6667e9-7573-41c4-b263-3fc4d4b88c93_mask.jpg")

    images_per_prompt = 20
    guidance_scale = 5
    num_inference_steps = 100
    prompts = ["Green volkswagen Golf", "Green volkswagen golf, detailed"]

    
    # Call pipeline.run as you wish
    for prompt in prompts:    
        for _ in range(5):
            inpaint_pipeline.run(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt="White, Grey, Deformed",
                guidance_scale=guidance_scale,
                images_per_prompt=images_per_prompt,
                num_inference_steps=num_inference_steps,
            )
    # for _ in range(0):
    #     text2img_pipeline.run(
    #         prompt="Photography of a red Volkswagen Golf 8",
    #         negative_prompt="White, Grey, Deformed",
    #         guidance_scale=guidance_scale,
    #         images_per_prompt=images_per_prompt,
    #         num_inference_steps=num_inference_steps,
    #         height=512,
    #         width=512
    #     )
    #
    # print("Hello!")