PyBoard
======

- [Quick Start](#quick-start)
- [Under-the-hood](#under-the-hood)
- [Contributing](#contributing)


# Quick Start

This section offers a brief overview with the intent of getting up and running in as short a time as possible. If your goal is to get up and running ASAP, this is the part for you.

## Setup

#### Imports

```
pip install transformers datasets evaluate
```

# Under-the-hood

## Available Models

The models available for us to work with were chosen to meet several criteria:

- Accessible locally
- Open-source
- Free to use
- Sufficiently capable

Apart from the last criteria, these basically boil down to that we want to be able to host the model on our local system. We don't want to make API calls to external models because, well, that typically costs money.

### [LLama](https://llama.meta.com/)

The first set of models available to use are the Meta AI LLMs, Llama. Specifically, Llama2, Llama3, and CodeLlama. To download the weights for these models, you will have to go to their site and accept their license. This wil give you a unique url you will have to pass to their download scripts. The actual scripts are included in the scripts directory here for convenience along with a [python version](scripts/llama_download/llama_download.py) I've written to make them OS-agnostic.

> Please save copies of the unique custom URLs provided above, they will remain valid for 24 hours to download each model up to 5 times, and requests can be submitted multiple times. An email with the download instructions will also be sent to the email address you used to request the models.

#### [Getting Started with Meta Llama](https://llama.meta.com/docs/get-started/)

> This guide provides information and resources to help you set up Llama including how to access the model, hosting, how-to and integration guides. Additionally, you will find supplemental materials to further assist you while building with Llama.

#### [Responsible Use Guide](https://llama.meta.com/responsible-use-guide/)

> The Responsible Use Guide is a resource for developers that provides best practices and considerations for building products powered by large language models (LLM) in a responsible manner, covering various stages of development from inception to deployment.

#### [Llama3](https://github.com/meta-llama/llama3)

The linked repo goes over a minimal example of loading Llama3 models and running inference. [This repo](https://github.com/meta-llama/llama-recipes) has a more expansive library of examples on how to run and use Llama3.

#### [Llama2](https://github.com/meta-llama/llama)

#### [CodeLlama](https://github.com/meta-llama/codellama)

## Learning Resources

- [Prompt Engineering](https://www.coursera.org/specializations/prompt-engineering)
- [Host Ai Locally - Easy Method for LLMs](https://www.youtube.com/watch?v=L12865sEB-o)
- [LM Studio Server](https://lmstudio.ai/docs/local-server)
- [Unleash the Power of Local Open Source LLM Hosting](https://yattishr.medium.com/unleash-the-power-of-local-open-source-llm-hosting-e33bf6a9679f)
- [5 easy ways to run an LLM locally](https://www.infoworld.com/article/3705035/5-easy-ways-to-run-an-llm-locally.html)
- [RayLLM - LLMs on Ray](https://github.com/ray-project/ray-llm)
- [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)
- [GGUF](https://huggingface.co/docs/hub/en/gguf)
- [koboldcpp](https://github.com/LostRuins/koboldcpp)
- [ollama](https://ollama.com/)
- [OLMo](https://github.com/allenai/OLMo)
- [OLMo-Eval](https://github.com/allenai/OLMo-Eval)
- [WildBench](https://github.com/allenai/WildBench)
- [Introducing BLOOM](https://bigscience.huggingface.co/blog/bloom)
- [Exploring BLOOM](https://www.datacamp.com/blog/exploring-bloom-guide-to-multilingual-llm)
- [BLOOM](https://huggingface.co/bigscience/)

## Huggingface

Core to this project is [huggingface](https://huggingface.co/docs/transformers/index). They are an open community historically specializing in reinforcement learning and natural language processing. They also host several competitions for open-source LLMs that we can use with little to no licensing restrictions. I highly recommend them as a starting point for getting a handle on some of the aspects of this domain. And even beyond the basics, they offer great resources for advanced topics as well as well-documented code and datasets we can use to bootstrap our models.

Admittedly, their [documentation](https://huggingface.co/docs) is *comprehensive*, which also means it's *long*. Rather than trying to understand everything, pick one or two pieces of interest and starting plugging away. Personally, I find the [datasets](https://huggingface.co/docs/datasets/index) to be rather accessible.

And if you're interested in learning the concepts to accompany the code, there is a [wealth of knowledge](https://huggingface.co/learn) on topics on [NLP](https://huggingface.co/learn/nlp-course), [deep learning](https://huggingface.co/learn/deep-rl-course), [computer vision](https://huggingface.co/learn/computer-vision-course), and even [ML for Games!](https://huggingface.co/learn/ml-games-course)

#### Imports

```
pip install transformers datasets evaluate
```

#### Tutorials

- [Data preprocessing](https://huggingface.co/docs/transformers/preprocessing)
- [Fine-tuning pre-trained model](https://huggingface.co/docs/transformers/training)
- [Scripting training](https://huggingface.co/docs/transformers/run_scripts)
- [Agents](https://huggingface.co/docs/transformers/agents)
- [Generation](https://huggingface.co/docs/transformers/llm_tutorial)
- [Chatting](https://huggingface.co/docs/transformers/conversations)

# Contributing
