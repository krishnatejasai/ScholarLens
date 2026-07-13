# ScholarLens Claim Verification Error Analysis

- Total evaluated claims: 60
- Incorrect predictions: 15

## c01

**Claim:** Dense Passage Retrieval uses separate learned encoders to represent questions and passages.

- Expected: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Support score: `0.1969`
- Contradiction score: `0.9948`
- Evidence paper: `paper_01_dpr`
- Page: `2`
- Chunk: `paper_01_dpr_page_2_chunk_6`

**Decisive evidence:**

> 3 Dense Passage Retriever (DPR) We focus our research in this work on improving the retrieval component in open-domain QA. Given a collection of M text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efﬁciently the top k passages relevant to the input question for the reader at run-time. Note that M can be very large (e.g., 21 million passages in our experiments, described in Section 4.1) and k is usually small, such as 20–100. 3.1 Overview Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a ddimensional real-valued vectors and builds an index f...

## c03

**Claim:** Contriever learns dense retrieval representations using contrastive learning without relevance labels.

- Expected: `SUPPORTED`
- Predicted: `INSUFFICIENT_EVIDENCE`
- Support score: `0.0059`
- Contradiction score: `0.4079`
- Evidence paper: `paper_01_dpr`
- Page: `9`
- Chunk: `paper_01_dpr_page_9_chunk_1`

**Decisive evidence:**

> Using labeled pairs of queries and documents, discriminatively trained dense encoders have become popular recently (Yih et al., 2011; Huang et al., 2013; Gillick et al., 2019), with applications to cross-lingual document retrieval, ad relevance prediction, Web search and entity retrieval. Such approaches complement the sparse vector methods as they can potentially give high similarity scores to semantically relevant text pairs, even without exact token matching. The dense representation alone, however, is typically inferior to the sparse one. While not the focus of this work, dense representations from pretrained models, along with cross-attention mechanisms, have also been shown effective i...

## c06

**Claim:** RAG combines parametric sequence-to-sequence memory with a non-parametric dense document index.

- Expected: `SUPPORTED`
- Predicted: `INSUFFICIENT_EVIDENCE`
- Support score: `0.4954`
- Contradiction score: `0.0132`
- Evidence paper: `paper_06_rag`
- Page: `2`
- Chunk: `paper_06_rag_page_2_chunk_1`

**Decisive evidence:**

> For query x, we use Maximum Inner Product Search (MIPS) to ﬁnd the top-K documents zi. For ﬁnal prediction y, we treat z as a latent variable and marginalize over seq2seq predictions given different documents. but have only explored open-domain extractive question answering. Here, we bring hybrid parametric and non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer to as retrieval-augmented generation (RAG). We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-para...

## c10

**Claim:** RAGAS provides reference-free metrics for evaluating retrieval-augmented generation pipelines.

- Expected: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Support score: `0.9846`
- Contradiction score: `0.9997`
- Evidence paper: `paper_10_ragas`
- Page: `1`
- Chunk: `paper_10_ragas_page_1_chunk_5`

**Decisive evidence:**

> While the usefulness of retrieval-augmented strategies is clear, their implementation requires a significant amount of tuning, as the overall performance will be affected by the retrieval model, the considered corpus, the LM, or the prompt formulation, among others. Automated evaluation of retrieval-augmented systems is thus paramount. In practice, RAG systems are often evaluated in terms of the language modelling task itself, i.e. by measuring perplexity on some reference corpus. However, such evaluations are not always predictive of downstream performance (Wang et al., 2023c). Moreover, this evaluation strategy relies on the LM probabilities, which are not accessible for some closed models...

## c11

**Claim:** ALCE studies how language models can generate answers with verifiable citations.

- Expected: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Support score: `0.9790`
- Contradiction score: `0.9999`
- Evidence paper: `paper_11_alce`
- Page: `9`
- Chunk: `paper_11_alce_page_9_chunk_2`

**Decisive evidence:**

> 7 Related Work Evaluating citations. Generating text with citations is closely related to attribution. Rashkin et al. (2023) define the “attributable to identified sources” (AIS) score to measure how faithful a generated text is to its sources. Bohnet et al. (2022) apply AIS scores on a single-document short-answer QA dataset. Honovich et al. (2022); Yue et al. (2023) study automatic evaluations for the AIS score. A concurrent work (Liu et al., 2023) conduct human evaluation on commercial generative search engines to examine their citation qualities. Scientific citation text generation (Funkquist et al., 2022) is a related task to ALCE where the model is provided the papers-to-cite and conte...

## c12

**Claim:** FActScore decomposes long-form generations into atomic facts for factual-precision evaluation.

- Expected: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Support score: `0.7872`
- Contradiction score: `0.9922`
- Evidence paper: `paper_09_crag`
- Page: `12`
- Chunk: `paper_09_crag_page_12_chunk_0`

**Decisive evidence:**

> Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 12076– 12100. Association for Computational Linguistics. Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. 2023. Generating benchmarks for factuality evaluation of language models. CoRR, abs/2307.06908. Deepak Narayanan, Mohammad Shoe...

## c16

**Claim:** SPECTER uses citation information to learn scientific document representations.

- Expected: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Support score: `0.9948`
- Contradiction score: `0.9987`
- Evidence paper: `paper_16_specter`
- Page: `1`
- Chunk: `paper_16_specter_page_1_chunk_5`

**Decisive evidence:**

> In this paper, we introduce a new method for learning general-purpose vector representations of scientiﬁc documents. Our system, SPECTER,2 incorporates inter-document context into the Transformer (Vaswani et al., 2017) language models (e.g., SciBERT (Beltagy et al., 2019)) to learn document representations that are effective across a wide-variety of downstream tasks, without the need for any task-speciﬁc ﬁne-tuning of the pretrained language model. We speciﬁcally use citations as a naturally occurring, inter-document incidental supervision signal indicating which documents are most related and formulate the signal into a triplet-loss pretraining objective. Unlike many prior works, at inferen...

## c20

**Claim:** Lost in the Middle reports weaker performance when relevant information appears in the middle of long contexts.

- Expected: `SUPPORTED`
- Predicted: `INSUFFICIENT_EVIDENCE`
- Support score: `0.1223`
- Contradiction score: `0.0006`
- Evidence paper: `paper_20_lost_in_middle`
- Page: `1`
- Chunk: `paper_20_lost_in_middle_page_1_chunk_5`

**Decisive evidence:**

> 1st 5th 10th 15th 20th Position of Document with the Answer 55 60 65 70 75 Accuracy 20 Total Retrieved Documents (~4K tokens) gpt-3.5-turbo-0613 gpt-3.5-turbo-0613 (closed-book) Figure 1: Changing the location of relevant information (in this case, the position of the passage that answers an input question) within the language model’s input context results in a U-shaped performance curve—models are better at using relevant information that occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performance degrades significantly when models must access and use information located in the middle of its input context. relevant documents from a search engine,...

## c30

**Claim:** RAGAS requires a gold reference answer for every metric it computes.

- Expected: `CONTRADICTED`
- Predicted: `INSUFFICIENT_EVIDENCE`
- Support score: `0.0023`
- Contradiction score: `0.6206`
- Evidence paper: `paper_10_ragas`
- Page: `1`
- Chunk: `paper_10_ragas_page_1_chunk_1`

**Decisive evidence:**

> RAG systems are composed of a retrieval and an LLM based generation module, and provide LLMs with knowledge from a reference textual database, which enables them to act as a natural language layer between a user and textual databases, reducing the risk of hallucinations. Evaluating RAG architectures is, however, challenging because there are several dimensions to consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithful way, or the quality of the generation itself. With Ragas, we put forward a suite of metrics which can be used to evaluate these different dimensions without having to rely on...

## c51

**Claim:** ALCE reports a controlled user study comparing citation usefulness for novice and expert readers.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0013`
- Contradiction score: `0.9430`
- Evidence paper: `paper_11_alce`
- Page: `15`
- Chunk: `paper_11_alce_page_15_chunk_2`

**Decisive evidence:**

> Table 11 demonstrates the experiments to show that ALCE is robust to shortcut cases. Using the top-1 passages or first two sentences of the top-1 passages induces almost perfect citation quality, but fluency and correctness are dramatically lower. E Citation Recall Discussion Our citation precision evaluation cannot detect a citation that partially supports the statement and hence will falsely penalize it. Consider a statement s3 and its citations [2][4][5]: if [2] entails partial information of s3 that [4][5] also entails, [2] will be counted as “irrelevant” while it should not be penalized. Liu et al. (2023) conduct human evaluation on citation precision in a different way: For each citati...

## c52

**Claim:** FActScore reports expected calibration error for its factuality predictions.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0105`
- Contradiction score: `0.9963`
- Evidence paper: `paper_12_factscore`
- Page: `4`
- Chunk: `paper_12_factscore_page_4_chunk_4`

**Decisive evidence:**

> More details are provided in Appendix A.3. 3.4 Results Statistics of the data and results are reported in Table 1. All LMSUBJs struggle with factual precision errors. InstructGPT and ChatGPT achieve FACTSCOREs of 42.5% and 58.3%, respectively. PerplexityAI, which uses a commercial search engine and thus should have a perfect FACTSCORE if directly copying the text from the correct Wikipedia page, attains a FACTSCORE of 71.5%. We provide a qualitative analysis of its error cases in the last paragraph of this section. ChatGPT and PerplexityAI often abstain from answering which presumably improves their factual precision. InstructGPT rarely abstains from answering, likely because it is not train...

## c53

**Claim:** SciFact records annotator confidence using a five-point confidence scale.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0005`
- Contradiction score: `0.9951`
- Evidence paper: `paper_14_scifact_open`
- Page: `4`
- Chunk: `paper_14_scifact_open_page_4_chunk_4`

**Decisive evidence:**

> The more systems identiﬁed a given CAP, the more likely it is to contain evidence. Total Retrieved Annotated ECAPs 209 187 (89%) 171 (82%) (c) Count of how many ECAPs from SCIFACT-ORIG would have been identiﬁed during pooled data collection. “Retrieved” indicates the number of ECAPs that would have been retrieved among the top k, and “Annotated” indicates the number that would further have been included in the annotation pool. Table 2: Annotation results and dataset statistics for SCIFACT-OPEN. than one system. The majority of CAPs are selected by a single system only, indicating high diversity in model predictions. As mentioned in §3.1, the total number of annotated CAPs is 732 (rather than...

## c56

**Claim:** SPECTER is evaluated on scientific author-disambiguation as a downstream task.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0033`
- Contradiction score: `0.9027`
- Evidence paper: `paper_16_specter`
- Page: `3`
- Chunk: `paper_16_specter_page_3_chunk_5`

**Decisive evidence:**

> 2.5 Inference At inference time, the model receives one paper, P, and it outputs the SPECTER’s Transfomer pooled output activation as the paper representation for P (Equation 1). We note that for inference, SPECTER requires only the title and abstract of the given input paper; the model does not need any citation information about the input paper. This means that SPECTER can produce embeddings even for new papers that have yet to be cited, which is critical for applications that target recent scientiﬁc papers. 3 SCIDOCS Evaluation Framework Previous evaluations of scientiﬁc document representations in the literature tend to focus on small datasets over a limited set of tasks, and extremely h...

## c57

**Claim:** SciNCL reports zero-shot retrieval results on patent documents.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0125`
- Contradiction score: `0.9978`
- Evidence paper: `paper_03_contriever`
- Page: `16`
- Chunk: `paper_03_contriever_page_16_chunk_4`

**Decisive evidence:**

> In Proceedings of the 23rd international conference on world wide web, pp. 373–374, 2014. 2 Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021. 2, 6, 18, 19 James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and veriﬁcation. arXiv preprint arXiv:1803.05355, 2018. 1 16

## c59

**Claim:** Longformer is evaluated on source-code summarization benchmarks.

- Expected: `INSUFFICIENT_EVIDENCE`
- Predicted: `CONTRADICTED`
- Support score: `0.0002`
- Contradiction score: `0.9998`
- Evidence paper: `paper_19_longformer`
- Page: `1`
- Chunk: `paper_19_longformer_page_1_chunk_1`

**Decisive evidence:**

> Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and ﬁnetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.
