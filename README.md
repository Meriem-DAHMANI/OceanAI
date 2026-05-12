# OceanAI 🌊
OceanAI is a domain specific chatbot built to answer questions about marine science, ocean ecosystems, and oceanography.  It combines two complementary 
approaches:
-  **Fine-tuned LLM** to handle foundational marine science knowledge  (biology, ecosystems, oceanography concepts)
- **RAG pipeline** to retrieve and grounds answers from recent research papers and scientific sources

This is an open-source project following an iterative development approach, each version improves through better data, techniques, and refinements

### Data
Built from open source marine content:
- **Wikipedia** (CC BY-SA 4.0) — marine biology & oceanography articles


### Datasets on Hugging Face
| Dataset | Purpose |
|---------|---------|
| [marine-dataset-qa](https://huggingface.co/datasets/Meriem-DH/marine-dataset-qa) | Q&A pairs for instruction finetuning |
| [marine-dataset-cpt](https://huggingface.co/datasets/Meriem-DH/marine-dataset-cpt) | Continued pre training on marine text |

More sources will be added in future versions


### Why ? 🌍 
Marine science knowledge is scattered across hundreds of sources and often hard to access. OceanAI brings it together and makes it easy to explore through conversation, for students, researchers, and ocean enthusiasts alike.

### License
Code: MIT — Dataset: CC BY-SA 4.0
