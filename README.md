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


## Discussion
Across a range of tasks, there appears to be an inverse relationship between a model’s calibration and its overall proficiency. As task complexity increases, models struggle to assign well-calibrated confidence scores to their predictions. Ideally, a model should express lower certainty when it is less well-suited to a particular task, mirroring a human's intuitive self-assessment. However, current language models often lack this adaptive uncertainty. Notably, in open-source models such as Llama, internal logit probabilities tend to align with their stated confidence, suggesting that the model possesses an implicit sense of certainty, even if it lacks an accurate awareness of task difficulty. Furthermore, task difficulty is exacerbated by an increase in the number of answer choices; for instance, empirical results show that GPT-4 exhibits lower expected calibration error (ECE) on tasks with fewer answer options, such as SciQ compared to the LSAT, indicating improved confidence alignment in simpler settings. Additionally, when explicitly prompted to report confidence levels, language models tend to emulate human tendencies by rounding their confidence scores to the nearest multiple of ten (e.g., 70% rather than 73.56%), reflecting learned patterns in probabilistic expression.

## Next Steps
Moving forward, we aim to investigate the relationship between task difficulty and model calibration, with particular attention to whether increased difficulty serves as a reliable indicator of how responsive a model’s confidence is to its actual accuracy. Unlike humans, who are often aware when they lack knowledge in a specific domain, large language models (LLMs) do not reliably exhibit such meta-cognitive awareness. This raises important concerns about the degree to which LLMs are well-calibrated in their confidence judgments.

Another key direction involves generating and analyzing controlled hallucinations. By systematically eliciting hallucinated responses, we hope to isolate and identify the underlying factors that contribute to these errors. This would facilitate a more precise understanding of the mechanisms behind hallucination and inform strategies for mitigation.

In parallel, we plan to conduct a more granular analysis of model performance across different types of questions—such as mathematical reasoning, logical inference, and reading comprehension—leveraging our labeled dataset to determine the specific domains in which LLMs perform reliably versus those where they are more prone to error.

Lastly, we seek to examine the models’ ability to differentiate between correct and incorrect answers. This line of inquiry will shed light on the extent to which current LLMs possess internal mechanisms for answer verification and whether these mechanisms can be refined to enhance both performance and trustworthiness.



## Additional Visualizations
<img width="1432" alt="image" src="https://github.com/user-attachments/assets/8b84e99e-d362-4b8b-9912-29b6ad5c6ad0" />
<img width="1437" alt="image" src="https://github.com/user-attachments/assets/d472dbc1-37d9-40f1-9238-ce57a580bb90" />
<img width="1431" alt="image" src="https://github.com/user-attachments/assets/e09439d6-422d-4bd1-ab1a-0c8d8a9e35a7" />
<img width="1433" alt="image" src="https://github.com/user-attachments/assets/c352e728-1961-42a3-bedc-99b1efacc2f8" />






