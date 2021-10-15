import requests, nltk, pymorphy2, json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from natasha import (
    Segmenter,
    MorphVocab,  
    PER,
    NewsMorphTagger,
    NamesExtractor,
    NewsNERTagger,   
    NewsEmbedding,
    Doc
)


# Текст статьи по ссылке (КиберЛенинки)
def getText(url):
    r = requests.get(url)
    bs = BeautifulSoup(r.text, 'html.parser') 

    textRaw = bs.find('div', {'class': 'ocr'})
    text = ''
    
    if (textRaw == None):
        print('Вероятно, КиберЛенинка хочет проверить Вас на робота!')
        return 'Что-то пошло не так!'
    for elem in textRaw:
        if (elem.text == 'iНе можете найти то, что вам нужно? Попробуйте сервис подбора литературы.'):
            continue
        text += elem.text
    return text

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
        "Accept": "*/*",
        "Accept-Language": "ru,en;q=0.5",
        "Content-Type": "application/json",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "max-age=0"
    }

#Максимальное количество статей (если столько найдено не будет, то придет, сколько есть)
length = 5

#Что ищем
q = "веб блокчейн"

data = {"mode":"articles","q":q,"size":length,"from":0}

response = requests.post('https://cyberleninka.ru/api/search', headers=headers, data=json.dumps(data))

res = json.loads(response.text)

# Массив ссылок со статьями 
urls = []
for article in res["articles"]:
    urls.append("https://cyberleninka.ru" + article['link'])
    
# Тексты найденных статей
texts = ''
for url in urls:
    texts += getText(url)

emb = NewsEmbedding()
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

doc = Doc(texts)
 
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.tag_ner(ner_tagger)
 
for span in doc.spans:
    span.normalize(morph_vocab)
normText = {_.text: _.normal for _ in doc.spans}

 
for span in doc.spans:
    if span.type == PER:
        span.extract_fact(names_extractor) 
fullNames = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}

# Только фамилии, потому как они всегда совпадают, а имена либо полностью, либо инициалы
# !Могут попасть слова с заглавной буквы, такие как Блокчейн, Веб (конкретно эти исключены)
lastNames = []
for name in fullNames.values():
    if (name.get('last') == None or len(name.get('last')) < 3 or name.get('last') == 'Веб' or name.get('last') == 'Блокчейн'):
        continue
    lastNames.append(name.get('last'))
    
# Ключевые персонажи, публикующие статьи
fdist = FreqDist(lastNames)
print(fdist.most_common(5))

# Предполагаемые термины (существительные)
terms = []
for token in doc.tokens:
    if ((token.pos == 'NOUN' or token.pos == 'PROPN' or token.pos == 'PART') and len(token.text) > 1):
        terms.append(token.text.lower())

# Лемматизация терминов
morph = pymorphy2.MorphAnalyzer()
lemmatizedTextsTokens = []

for word in terms:
    p = morph.parse(word)[0]
    lemmatizedTextsTokens.append(p.normal_form)
    
# Фильтрация лишних слов, которые могли попасть в термины
# Можно добавить общие слова (например, 'данные', 'решение', 'задача' и т.д.)
badwords = stopwords.words("russian")
badwords.extend(stopwords.words("english"))
badwords.extend(['doi', 'страница'])

filteredTextsTokens = []

for word in lemmatizedTextsTokens:
    if word not in badwords:
        filteredTextsTokens.append(word)

# 30 популярных терминов
termsText = nltk.Text(filteredTextsTokens)
fdist = FreqDist(termsText)
print(fdist.most_common(30))

# tag cloud по терминам
termsRaw = " ".join(termsText)
wordcloud = WordCloud().generate(termsRaw)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()