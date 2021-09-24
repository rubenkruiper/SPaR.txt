# SPaR.txt - a cheap Shallow Parsing approach for Regulatory texts
_This work was published in the EMNLP 2021 workshop on [Natural Legal Language Processing](http://nllpw.org/)_    

> **Abstract**: Automated Compliance Checking (ACC) systems aim to semantically parse building regulations to a set of rules. 
However, semantic parsing is known to be hard and requires large amounts of training data. 
The complexity of creating such training data has led to research that focuses on small sub-tasks, 
such as shallow parsing or the extraction of a limited subset of rules. This study introduces a shallow parsing 
task for which training data is relatively cheap to create, with the aim of learning a lexicon for ACC. 
We annotate a small domain-specific dataset of 200 sentences, SPaR.txt, 
and train a sequence tagger that achieves 79,93 F1-score on the test set. We then show through 
manual evaluation that the model identifies most (89,84\%) defined terms in a set of building regulation 
documents, and that both contiguous and discontiguous Multi-Word Expressions (MWE) are discovered with 
reasonable accuracy (70,3\%).

This repository contains:
* ScotReg - a json corpus of the [domestic](https://www.gov.scot/publications/building-standards-technical-handbook-2020-domestic/) and [non-domestic](https://www.gov.scot/publications/building-standards-technical-handbook-2020-non-domestic/) Scottish Building Regulations 
   * scraped 14th of June 2021 
   * please also see[crown copyright](https://www.gov.scot/crown-copyright/) and the [Open Government License](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
* 200 randomly selected sentences annotated using [BRAT](https://brat.nlplab.org/), see [the SpaR.txt paper]() for annotation details
* Code to train a biLSTM+CRF tagger, in support of learning a lexicon (including MWE discovery/identification)

**To Do**: 
* I aim to provide some details on how to train, predict, and evaluate here
* I'll update the details on our paper

If you use code or data from SPaR.txt in your research, please consider citing the following papers:
```
@inproceedings{Kruiper2021_SPaRtxt,
  author =      "Kruiper, Ruben
                and Konstas, Ioannis
                and Gray, Alasdair J,
                and Sadeghineko, Farhad,
                and Watson, Richard,
                and Kumar, Bimal",
  title =       "SPaR.txt, a cheap Shallow Parsing approach for Regulatory texts"
  year =        "2021",
  url =         "https://github.com/rubenkruiper/SPaR.txt",
}
```
The code and BRAT annotations in this repository are licensed under a Creative Commons Attribution 4.0 License.
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-sa.png" width="134" height="47">