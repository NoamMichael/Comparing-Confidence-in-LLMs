# Comparing-Confidence-in-LLMs
Large Language Model Overconfidence Project within Moore Accuracy Lab at Berkeley Haas <br>
_Authors: Noam Michael, Kelly Hu, Daniel BenShushan_ <br>
_Advised By: Advised by Prof. Don Moore & Prof. Jacob Bien_<br>
_In Collaboration with USC Marshall and Berkeley Haas_<br>

See the link to our poster here: https://drive.google.com/file/d/1Y8K6BILDCSWpU2C1oj10JocpeBsS6-So/view?usp=sharing

## Introduction
Large Language Models (LLMs) like GPT, Claude and Llama have become increasingly prevalent, but they often generate inaccurate information or "hallucinations," undermining their reliability. As these models are becoming more ubiquitous, it is becoming more necessary to detect and mitigate these inaccuracies.  To address this issue, users should be armed with knowledge of a model’s overall proficiency and with its ability to self assess its answers. Borrowing from meteorology, we can analyze a model’s performance in the context of calibration. A well-calibrated model, for instance, would be correct 80% of the time when it expresses 80% confidence. We measure model confidence in the form of its Stated Probability as well as the probability it assigns to each answer token (Logit Probability. We compare models by assessing their respective Expected Calibration Error (ECE). 

## Methodology
Our methodology utilizes a custom, standardized system prompt specifically designed for the SciQ and LSAT datasets. We implement a double prompting technique. In the first stage, the model is prompted to produce both answer and reasoning, This output is then integrated into a second prompt that incorporates the original question. The second prompt requests the model to provide its stated confidence for each answer option based on the initial reasoning. The resulting data is then aggregated into a table, from which we extract the probability assigned to each individual multiple choice option. These individual probabilities are grouped into confidence bins, enabling us to analyze both accuracy and calibration across all probability levels. Through analysing each individual option, we are able to have a more nuanced evaluation of the model’s performance.


## Measuring Confidence
**Stated Confidence** - This is defined as what the model believes its confidence for each answer choice is in a number ranging from 0.0 to 1.0. This is the number a model outputs when prompted "Provide the likelihood that each answer is correct (A, B, C, D)." 

**Logit Confidence** - Large Language Models produce outputs in the form of “tokens”, where the next token outputted is decided based on a probability, and is a measuring of how likely the next token will be predicted. For a given question, we took the Logit of the answered token divided by the sum of the logits across the answer space. As GPT is not open source, we were unable to obtain the correct logits consistently.

**Brier Score**:  This metric calculates the weighted average error of the estimated “probabilities” thus resulting in a single value that we can use to compare different models. Essentially, we are taking the squared difference of the accuracy and confidence. 

## Benchmarks Used
We tested Claude Sonnet 3.7, Llama-3.1 and GPT-4 on three different benchmarks:<br>
**LSAT** – Tests logical thinking and puzzles, requiring the model to understand the data and perform critical thinking to fully formulate an answer. Consists of a sample of 1567 LSAT questions (for this presentation, we used a sample of 200) from approximately 90 LSAT exams administered between  1991 and 2016. 

**SciQ** – The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics, Chemistry and Biology, among others. The questions are in multiple-choice format with 4 answer options each. For the majority of the questions, an additional paragraph with supporting evidence for the correct answer is provided.


