import re, string, pickle, json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SemanticRolesOptions, KeywordsOptions, CategoriesOptions

pklFile = open('09_03', 'rb')      
nineMarch = pickle.load(pklFile) 
combines = list(set(nineMarch.text))
# len(combines)

combined_text = ' '.join(combines).lower()
text = re.sub('\[.*?\]', '', text)
text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
text = re.sub('\w*\d\w*', '', text)
text = re.sub('[‘’“”…]', '', text)
text = re.sub('\n', '', text)
text = re.sub('[‘’“”…]', '', text)
text = re.sub('\n', '', text)
text = text.encode('ascii', 'ignore').decode('ascii')

authenticator = IAMAuthenticator('key')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    authenticator=authenticator
)

natural_language_understanding.set_service_url("{url}")

response = natural_language_understanding.analyze(
    text=text,
    features=Features(categories=CategoriesOptions(limit=3))).get_result()  
print(json.dumps(response, indent=2))
