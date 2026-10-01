# Attributions

## GermanWordEmbeddings

Data files in the directory "devmount" were taken from project [GermanWordEmbeddings](https://github.com/devmount/GermanWordEmbeddings), Copyright (c) 2015 Andreas Müller. These files are licensed under the MIT license. See DEVMOUNT-LICENSE.md for additional details.

## One Million Posts

"One Million Posts" Corpus (`evaluation_data/classification/million_posts_sentiment.feather`) by Schabus et al (2017) available at: [https://ofai.github.io/million-post-corpus/](https://ofai.github.io/million-post-corpus/). Redistributed granted by Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (see: `MillionPosts-LICENSE.md`).

### Citation

- Dietmar Schabus, Marcin Skowron, Martin Trapp. One Million Posts: A Data Set of German Online Discussions. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pp. 1241-1244. Tokyo, Japan, August 2017. DOI: [10.1145/3077136.3080711](https://doi.org/10.1145/3077136.3080711)
- Dietmar Schabus and Marcin Skowron. Academic-Industrial Perspective on the Development and Deployment of a Moderation System for a Newspaper Website. Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018), pp. 1602-1605. Miyazaki, Japan, May 2018. [Full paper available for download from LREC](http://www.lrec-conf.org/proceedings/lrec2018/summaries/8885.html)

# Descriptives

For all classification tasks we took a random split (75% for training, 25% for testing data) with a fixed random seed (`1234`).

## AUTNES 2017 (Multi-label classification)

Model gets text and needs to predict the topics / issues

Total N = 854

|label                |   N| proportion|
|:--------------------|---:|----------:|
|topic_society        | 127|       0.15|
|topic_infrastructure | 136|       0.16|
|topic_foreignpolicy  |  58|       0.07|
|topic_socialwelfare  | 103|       0.12|
|no label             | 507|       0.59|

## AUTNES 2019 (Multi-label classification)

Model gets text and needs to predict the topics / issues

total N = 948

|label                |   N| proportion|
|:--------------------|---:|----------:|
|topic_budget         | 233|       0.25|
|topic_socialwelfare  | 159|       0.17|
|topic_kultur         | 152|       0.16|
|topic_heer           | 173|       0.18|
|topic_aussenpolitik  |  62|       0.07|
|topic_europa         | 182|       0.19|
|topic_infrastructure | 140|       0.15|
|topic_society        | 188|       0.20|
|topic_environment    | 204|       0.22|
|topic_institutions   | 245|       0.26|
|topic_immigration    | 127|       0.13|
|topic_labout         | 110|       0.12|
|topic_conflict       | 403|       0.43|
|topic_games          | 393|       0.41|
|topic_euref          |  71|       0.07|
|topic_socialmedia    |  81|       0.09|
|topic_fakenews       |  46|       0.05|
|topic_polling        |  64|       0.07|
|topic_scandals       | 239|       0.25|
|no label             |  50|       0.05|


## AUTNES Sentiment Dataset

Model gets text and needs to predict the sentiment category

Total N = 4599

|label    |    n|
|:--------|----:|
|Negative | 1299|
|Neutral  | 2113|
|Positive | 1187|

## Facebook Party Messages

Fictious prediction task, model gets text as input and needs to guess the party.

Total N = 105,300

|label |     n|
|:-----|-----:|
|FPOE  | 47371|
|GRUE  | 11317|
|NEOS  | 11245|
|OEVP  | 10477|
|SPOE  | 24890|

## Austrian Parliament Speeches

Fictious prediction task, model gets text as input and needs to guess the party.

Total N = 25,075

|label |    n|
|:-----|----:|
|FPOE  | 5068|
|GRUE  | 2950|
|NEOS  | 2150|
|OEVP  | 7273|
|SPOE  | 7634|

## Press Releases by Austrian Parties

Fictious prediction task, model gets text as input and needs to guess the party.

Total N = 150,235

|label |     n|
|:-----|-----:|
|FPOE  | 32294|
|GRUE  | 14944|
|NEOS  |  7368|
|OEVP  | 54432|
|SPOE  | 41197|

## Twitter Posts by Austrian Parties

Fictious prediction task, model gets text as input and needs to guess the party.

Total N = 247,473

|label |      n|
|:-----|------:|
|FPOE  |   3182|
|GRUE  |  55429|
|NEOS  | 102183|
|OEVP  |  31796|
|SPOE  |  54883|
