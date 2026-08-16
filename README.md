# Applied ML Engineer Challenge

![logo](img/logo.png)

Hey there! Are you interested in LLMs? Do you like building real products, experimenting with neural networks, implementing different ideas and testing them out? Would you like to do that for a living? Then you're in the right place!
This is an official test for people interested in joining [BottleCapAI](https://www.bottlecapai.com).

---

## Objective

Design and implement a Machine Learning service capable of fixing newline placement in English natural language text. Develop a model,
and a service with a HTTP API. You can choose any architecture and method of obtaining the model, and any Python framework for the HTTP API. 
It is not necessary to aim for state-of-the-art performance, but the service should efficiently generate reasonable answers.

Examples:  
```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
``` 
-> 
```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.
[...]
```

For your solution, it is required to:
- Implement an API with at least one endpoint that accepts a text input and returns a text with fixed newlines.
- Provide a Dockerfile with the environment to run the service.
- Write tests for the service.
- Compute appropriate metrics for evaluating the performance of the model.
- Provide a `report.md` with instructions on how to run the service, explaining your approach, decisions taken, and reporting the results.
- [optional] Ideally, we would encourage you to provide us with a link to the service deployed, for instance on [Huggingface Spaces](https://huggingface.co/spaces), with a minimal UI
for testing the model (can be in [Streamlit](https://github.com/streamlit/streamlit) or basic HTML + js). Provide the space link in your report.

In case you need access to GPUs for developing your model, we recommend using free online solutions, such as [Google Colab](https://colab.research.google.com/),
[Modal notebooks](https://modal.com/products/notebooks), or [Kaggle kernels](https://www.kaggle.com/kernels). 
It is fine to focus on smaller models in case of any hardware-related difficulties, as we are not expecting SOTA performance.

---

## What's the point?

We are interested in your drive, interest in ML and ability to create useful working products with it, adapting models to certain tasks, 
skill in developing, evaluating, monitoring, and deploying production ready ML services, and general programming skills.

---

## Submission

To submit your results, run:
```bash
git bundle create <first name>-<last name>.bundle --all
```
Then send us your .bundle file to hey(at)bottlecapai.com with subject in format: \<first name\>-\<last name\>-applied-ml-test\>.

At this moment, we are interested mainly in candidates willing to relocate to Prague and authorized to work in the EU. (If you are an exceptional fit, we are happy to discuss possible support options).

---
## 📌 About BottleCapAI

At **BottleCapAI**, we’re making large language models **radically more efficient** — aiming for **100× improvements** over today’s approaches. 🚀  

### 👥 Founders
- Tomas Mikolov – creator of *word2vec*, pioneer of neural language models.  
- Jaroslav Beck – co-founder of *Beat Games* (*Beat Saber*, 10M+ copies sold, acquired by Meta).  
- David Herel – creator of Thinking Tokens, co-founder of an AI trading startup, and Amazon Alexa Prize finalist.

### 🌍 Our vision
Training frontier LLMs costs **tens of millions** today. Our new algorithms already cut that by **~50%** — and we’re just getting started. We’re building a European hub to push AI forward through **algorithms, not brute force**.  
 
📧 **hey(at)bottlecapai.com** · 🌐 [bottlecapai.com](https://www.bottlecapai.com)  
