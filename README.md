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
#### 2.1 Bag of Words


 

