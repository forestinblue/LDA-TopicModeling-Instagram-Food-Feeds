# NLP
## Instagram_food_contents(Hashtag, Review) to TOPIC MODELING(LDA)

### 1. Data Preprocessing
I collected data : Feed Contents(feed review, hashtag) of Korean Eat star gramers 
![1](https://user-images.githubusercontent.com/90318043/156325954-fba29545-4dcf-4c57-baf8-a6c7e63084b8.jpg)

> [information about KoNLPy  Mecab](https://konlpy-ko.readthedocs.io/ko/v0.4.3/api/konlpy.tag/)
Text Tokenization

```python
def clean_text(text):
    text = text.replace(".", "").strip()
    text = text.replace(",", " ").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
    text = re.sub(pattern=pattern, repl='', string=text)
    return text
    
def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    nouns = [s for s, t in tagged if t in ['NNG', 'NNP', 'VA', 'XR'] and len(s) >1]
    return nouns

def tokenize(df):
    tokenizer = Mecab(dicpath=r'C:/mecab/mecab-ko-dic')
    processed_data = []
    for sent in tqdm(df['text']):
        sentence = clean_text(str(sent).replace("\n", "").strip())
        processed_data.append(get_nouns(tokenizer, sentence))
    return processed_data
    
def save_processed_data(processed_data):
    with open("tokenized_data_", 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)      
            
 if __name__ == '__main__':
    df = pd.read_csv("C:/Users/junseok/Downloads/instagram_crawling.csv")
    df.columns=['id', 'text', 'hashtag']
    df.dropna(how='any')
    processed_data = tokenize(df)
    save_processed_data(processed_data)
```

### 2. LDA Topic Modeling
>[LDA](https://www.mygreatlearning.com/blog/understanding-latent-dirichlet-allocation/)

LDA assumes that documents are composed of words that help determine the topics and maps documents to a list of topics by assigning each word in the document to different topics.
#### 2.1 Bag of Words
![bow01](https://user-images.githubusercontent.com/90318043/156478164-98807c0a-b9d4-410e-acca-c6ef139d8fcc.jpg)

Bag of Words is a representation of text that describe  the occurrence of words within a document. We just keep traak of word counts and disregard the grammatical details and the word order.
 

#### 2.2 Modeling
[genism.ldamodel's patameter](https://radimrehurek.com/gensim/models/ldamodel.html)
>[genism.ldamodel's patameter(korean version)](https://coredottoday.github.io/2018/09/17/%EB%AA%A8%EB%8D%B8-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D/)
- num_topics: The number of requested latent topics to be extracted from the training corpus.
- chunksize: Number of documents to be used in each training chunk.
- passes: Number of passes through the corpus during training.
- iterations:  Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
- eval_every: Log perplexity is estimated every that many updates. 
- id2word: Mapping from word IDs to words

### 3. Visualization

>[what is pyLDAvis's λ (korean version)](https://lovit.github.io/nlp/2018/09/27/pyldavis_lda/)
 ![image](https://user-images.githubusercontent.com/90318043/156956579-d49578a9-089b-4933-938b-207072ccf4d6.png)
