# PSAT

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


# PHQ9 description + 3 additional questions

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

### 3 Added labels on top of PHQ9
10) Talking disease symptoms or diagnosis
11) Talking about antidepressant
12) Talking about family or relatonship

[PHQ9 depression ontology](https://github.com/AnonymousUserGithub/DepressionDetection/blob/main/PSAT%20Results/PHQ9%20depression%20ontology.csv) contains all 12 labels and phrases extracted using keybert with pos tags and keybart.


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

This collection can only be used for research purposes using the following [user agreement](https://tec.citius.usc.es/ir/code/eRisk2018_agreement.odt) and sending it to david.losada@usc.es . Visit [eRISK webpage](https://tec.citius.usc.es/ir/code/eRisk2022.html) for more details.



# PSAT

All implementations related to work available at : [Code](https://github.com/AnonymousUserGithub/DepressionDetection/tree/main/PSAT%20Results/Code)

[PSAT Examples](https://github.com/AnonymousUserGithub/DepressionDetection/blob/main/Examples%20for%20PSAT-1.pdf). 

[Detailed PSAT examples with all PHQ9 labels](https://github.com/AnonymousUserGithub/DepressionDetection/blob/main/MoreExamplesWithPSAT.pdf).

[More Attention visualizations for CLEF and PRIMATE Dataset snapshots](https://github.com/AnonymousUserGithub/DepressionDetection/tree/main/PSAT%20Results/Attention%20Visulaization%20Experiment%20Snapshots)
