# PSAT

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

# Software
-  Programming Language: Python
-  Libraries :
  -  PyTorch: This is used to build and train the neural network models.
  -  Transformers (Hugging Face): This is used to leverage pre-trained models like BERT and customize them for PSAT.
  -  NumPy: For numerical computations.
  -  pandas: For data manipulation and analysis.
  -  scikit-learn: This is for evaluation metrics and additional preprocessing.
  -  NLTK: This is for natural language processing tasks, including part-of-speech tagging.
  -  SNOMED-CT API: This is used to access clinical terminology and validate keyphrases.
  -  KeyBERT: For keyphrase extraction.

# Hardware
-  GPUs: Two NVIDIA Tesla V100 GPUs (16GB each) will be used to train the models.
-  CPUs: Intel Xeon processors for general computation tasks.
-  Memory: 128GB RAM to handle large datasets and model training processes.
-  Storage: SSD storage for faster data access and processing.



# Instructions To Run Py files

[Find Instructions here to run code](https://github.com/AnonymousUserGithub/DepressionDetection/blob/b52ea35272a2d339cb25c76a74c904cde5615d96/instructions.md)

# PHQ9 description 

### PHQ-9 labels
1) Little interest or pleasure in doing things
2) Feeling down, depressed, or hopeless
3) Trouble falling or staying asleep, or sleeping too much
4) Feeling tired or having little energy
5) Poor appetite or overeating	
6) Feeling bad about yourself—or that you are a failure or have let yourself or your family down
7) Trouble concentrating on things, such as reading the newspaper or watching television
8) Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual
9) Thoughts that you would be better off dead or of hurting yourself in some way


[PHQ9 depression ontology](PHQ9Questions_depression_ontology.csv) contains all 9 labels and phrases extracted using keybert with pos tags and keybart.


# Primate Dataset

The dataset is available for research purposes under proper user agreements. The [PRIMATE agreement document](https://github.com/primate-mh/Primate2022/blob/main/Primate2022_agreement.pdf) provides all details needed to have access to the PRIMATE 2022 dataset.

### Sample post:

```json
{
    "post_title": "Should I use the psychological help service that my university provides for free?",
    "post_text": "Lately I've been feeling really low. \nI can't make myself leave the bed, I start crying out of the blue and everything is just so heavy. \nI think I've always suffered from some kind of depression but I've never been to therapy because I couldn't afford it on my own and my family didn't ever suspect anything. \nNow I live on my own in another city. Yesterday I discovered that my university provides psychological help for students for free. Do you think I should give it a go? \nI'm a bit afraid because I don't know what to expect and I don't really know what to tell them when I'll be there. I know they don't provide help for very serious issues (you'll need a psychiatrist for that) and I hope they don't take care for only \"university related problems\".\nOn the other hand, I have nothing to lose because it's free.\nDid you ever try anything like that? \n",
    "annotations": [
      [
        "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down",
        "yes"
      ],
      [
        "Feeling-down-depressed-or-hopeless",
        "yes"
      ],
      [
        "Feeling-tired-or-having-little-energy",
        "yes"
      ],
      [
        "Little-interest-or-pleasure-in-doing ",
        "yes"
      ],
      [
        "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual",
        "no"
      ],
      [
        "Poor-appetite-or-overeating",
        "no"
      ],
      [
        "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way",
        "no"
      ],
      [
        "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television",
        "no"
      ],
      [
        "Trouble-falling-or-staying-asleep-or-sleeping-too-much",
        "yes"
      ]
    ]
  }
```
Detailed examples and information on PRIMATE dataset avilable at (https://github.com/primate-mh/Primate2022) .

# CLEF Dataset

This collection can only be used for research purposes using the following [user agreement](https://tec.citius.usc.es/ir/code/eRisk2018_agreement.odt) and sending it to david.losada@usc.es . Visit [eRisk webpage](https://tec.citius.usc.es/ir/code/eRisk2022.html) for more details.

# CAMS

The CAMS dataset have classes, 'No reason' (C0), 'Bias or abuse' (C1), 'Jobs and careers' (C2), 'Medication' (C3), 'Relationship' (C4), and 'Alienation' (C5). The dataset is presented in a <text, cause, inference> format, where ‘text’ is the user post with 'cause' referring to the labeled reason behind the mental disorder mentioned in the post, and ‘inference’ indicates the key phrases (or relevant concepts) in the post that expert annotators considered when assigning labels. For access to CAMS, please contact the author: Dr. Muskan Garg (muskanphd@gmail.com). Due to the presence of sensitive material, access to the content can be obtained via email.

CAMS distinguishes itself from CLEF e-Risk by offering ground truth explanations. In the subsequent examples, we verify whether the highlighted words/phrases in ClinicalT5 and PSAT coincide with the ground truth explanations provided by CAMS:
![Alt text of the image](https://github.com/AnonymousUserGithub/DepressionDetection/blob/3404a53d65a0ff0eb394cf5e21aa29c59dc735e9/Examples%20for%20PSAT.jpg)

Genration from ChatGPT

![Alt text](https://github.com/AnonymousUserGithub/DepressionDetection/blob/8ee3dd64bbb819221a2a69767937639312798fdf/gptexplation.svg
)
A Scenario where PSAT underperforms ClinicalT5 attention, but because of focused attention, the generated explanations were comparable:
https://github.com/AnonymousUserGithub/DepressionDetection/blob/16762ae6252f6cff657c62d05e392fbb7bc37c0a/exp2camsgpt.svg
![Alt text of the image](https://github.com/AnonymousUserGithub/DepressionDetection/blob/2852a5de0506ba2dcb98a5502a45338e5be9c920/CAMS1.png)

Genration from ChatGPT:

![Alt text of the image](https://github.com/AnonymousUserGithub/DepressionDetection/blob/16762ae6252f6cff657c62d05e392fbb7bc37c0a/exp2cams.svg)

# PSAT

All implementations related to work available at : [Code](https://github.com/AnonymousUserGithub/DepressionDetection/tree/main/PSAT%20Results/Code)

[PSAT Examples](https://github.com/AnonymousUserGithub/DepressionDetection/blob/main/Examples%20for%20PSAT-1.pdf). 

[Detailed PSAT examples with all PHQ9 labels](https://github.com/AnonymousUserGithub/DepressionDetection/blob/main/MoreExamplesWithPSAT.pdf).

[More Attention visualizations for CLEF and PRIMATE Dataset snapshots](https://github.com/AnonymousUserGithub/DepressionDetection/tree/main/PSAT%20Results/Attention%20Visulaization%20Experiment%20Snapshots)

