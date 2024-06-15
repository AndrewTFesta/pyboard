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

## Learning Resources

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
