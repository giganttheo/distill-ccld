## Distill CLOOB-Conditioned Latent Diffusion trained on WikiArt

[![](https://github.com/huggingface/community-events/raw/main/huggan/assets/huggan_banner.png?raw=true)](https://github.com/huggingface/community-events/tree/main/huggan)

As part of the [HugGAN](https://github.com/huggingface/community-events/tree/main/huggan) community event, I trained a 105M-parameters latent diffusion model using a knowledge distillation process.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giganttheo/distill-ccld/blob/master/distillCCLD_(Wikiart)_demo.ipynb) [![](https://img.shields.io/badge/W%26B-Report-yellow)](https://wandb.ai/gigant/distill-ccld/reports/Distill-Diffusion-105M--VmlldzoxODQwMTUz?accessToken=mfbrz1ghfakmh01lybsuycwm3qj3isv60uynnvmina3tiwz5e5ufbjui5xqhmaqi)
[![](https://img.shields.io/badge/%F0%9F%A4%97-Hub-yellow)](https://huggingface.co/huggan/distill-ccld-wa)
[![](https://img.shields.io/badge/%F0%9F%A4%97-Space-yellow)](https://huggingface.co/spaces/huggan/wikiart-diffusion-mini)


<p align="center">
    <img src="https://github.com/giganttheo/distill-ccld/blob/master/distill-snow.png" alt="drawing" width="500"/>
    <div align="center">Prompt : "A snowy landscape, oil on canvas"</div>
</p>


#### Links

* [Model card for the teacher model on HuggingFace](https://huggingface.co/huggan/ccld_wa), trained by Jonathan Whitaker. He described the model and training procedure on his [blog post](https://datasciencecastnet.home.blog/2022/04/12/fine-tuning-a-cloob-conditioned-latent-diffusion-model-on-wikiart/)
* [Model card for the student model on HuggingFace](https://huggingface.co/huggan/distill-ccld-wa), trained by me. You can check my [WandB report](https://wandb.ai/gigant/distill-ccld/reports/Distill-Diffusion-105M--VmlldzoxODQwMTUz?accessToken=mfbrz1ghfakmh01lybsuycwm3qj3isv60uynnvmina3tiwz5e5ufbjui5xqhmaqi). This version has 105M parameters, against 1.2B parameters for the teacher version. It is lighter, and allows for faster inference, while maintaining some of the original model capability at generating paintings from prompts.
* [Gradio demo app on HuggingFace's Spaces](https://huggingface.co/spaces/huggan/wikiart-diffusion-mini) to try out the model with an online demo app
* [iPython Notebook](https://github.com/giganttheo/distill-ccld/blob/master/distillCCLD_(Wikiart)_demo.ipynb) to use the model in Python
* [WikiArt dataset on `datasets` hub](https://huggingface.co/datasets/huggan/wikiart)
* [GitHub repository](https://github.com/giganttheo/distill-ccld)



#### How to use

You need some dependancies from multiple repositories linked in this repository : [CLOOB latent diffusion](https://github.com/JD-P/cloob-latent-diffusion) :

* [CLIP](https://github.com/openai/CLIP/tree/40f5484c1c74edd83cb9cf687c6ab92b28d8b656)
* [CLOOB](https://github.com/crowsonkb/cloob-training/tree/136ca7dd69a03eeb6ad525da991d5d7083e44055) : the model to encode images and texts in an unified latent space, used for conditioning the latent diffusion.
* [Latent Diffusion](https://github.com/CompVis/latent-diffusion/tree/f13bf9bf463d95b5a16aeadd2b02abde31f769f8) : latent diffusion model definition
* [Taming transformers](https://github.com/CompVis/taming-transformers/tree/24268930bf1dce879235a7fddd0b2355b84d7ea6) : a pretrained convolutional VQGAN is used as an autoencoder to go from image space to the latent space in which the diffusion is done.
* [v-diffusion](https://github.com/crowsonkb/v-diffusion-pytorch/tree/ffabbb1a897541fa2a3d034f397c224489d97b39) : contains some functions for sampling using a diffusion model with text and/or image prompts.

An example code to use the model to sample images from a text prompt can be seen in a [Colab Notebook](https://colab.research.google.com/drive/1XGHdO8IAGajnpb-x4aOb-OMYfZf0WDTi?usp=sharing), or directly in the [app source code](https://huggingface.co/spaces/huggan/wikiart-diffusion-mini/blob/main/app.py) for the Gradio demo on [this Space](https://huggingface.co/spaces/huggan/wikiart-diffusion-mini)


### Demo images

<p align="center">
    <img src="https://github.com/giganttheo/distill-ccld/blob/master/distill-mars.png" alt="drawing" width="500"/>
    <div align="center">Prompt : "A martian landscape painting, oil on canvas"</div>
</p>
