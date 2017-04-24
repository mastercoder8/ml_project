#Testing En- Fr
import sys
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models as word2vec
print('-----------------------------------------------------')
print('Word2Vec: [Loading]')
model_en = word2vec.Word2Vec.load('englishwords')
model_fr = word2vec.Word2Vec.load('frenchwords')

data_en = open('test.en','r',encoding='utf8').read()
data_en = data_en.split()

words_en = list(set(data_en))
data_size_en,vocab_size_en = len(data_en),len(model_en.wv['the'])

print('-----------------------------------------------------')
print('English Corpus file has %d words,%d unique'%(data_size_en,len(words_en)))

word_to_ix_en = {}
ix_to_word_en = {}


for w in words_en:
    word_to_ix_en[w] = model_en.wv[w]
    ix_to_word_en[tuple(model_en.wv[w])] = w
print('English unique words = %d, word2vector features = %d'%(len(word_to_ix_en),len(model_en.wv['the'])))


print('-----------------------------------------------------')
lines_fr = open('test.fr','r',encoding='utf8').readlines()
data_fr = open('test.fr','r',encoding='utf8').read()
data_fr = data_fr.split()
words_fr = list(set(data_fr))
data_size_fr,vocab_size_fr = len(data_fr),len(model_en.wv['la'])
print('French  Corpus file has %d words,%d unique words'%(data_size_fr,len(words_fr)))
word_to_ix_fr = {}
ix_to_word_fr = {}

for w in words_fr:
    word_to_ix_fr[w] = model_fr.wv[w]
    ix_to_word_fr[tuple(model_fr.wv[w])] = w
print('French unique words =',len(word_to_ix_fr))
print('-----------------------------------------------------')
print('Word2Vec: [Loaded]')
print('-----------------------------------------------------')
print("Weights: [Loading]",end='')

#hyperparameters which are same for both encoder and decoder
hidden_size = 100
learning_rate = 1e-1


#encoder weight parameters for english language
wxh_en = np.random.randn(hidden_size,vocab_size_en)*0.01
whh_en = np.random.randn(hidden_size,hidden_size)*0.01
why_en = np.random.randn(vocab_size_en,hidden_size)*0.01
bh_en = np.zeros((hidden_size,1))
by_en = np.zeros((vocab_size_en,1))

#decoder weight parameters for french language
wxh_fr = np.random.randn(hidden_size,vocab_size_fr)*0.01
whh_fr = np.random.randn(hidden_size,hidden_size)*0.01
why_fr = np.random.randn(vocab_size_fr,hidden_size)*0.01
bh_fr = np.zeros((hidden_size,1))
by_fr = np.zeros((vocab_size_fr,1))


## Loading Weights
data = np.load('weights.en.npz')
wxh_en = data['wxh_en']
whh_en = data['whh_en']
why_en = data['why_en']
bh_en = data['bh_en']
by_en = data['by_en']

data = np.load('weights.fr.npz')
wxh_fr = data['wxh_fr']
whh_fr = data['whh_fr']
why_fr = data['why_fr']
bh_fr = data['bh_fr']
by_fr = data['by_fr']
print("\rWeights: [Loaded]");
## End Load Weights
def close():
	print('[EN] Thank you => Merci [FR]')
	sys.exit()
def cosine_dist(ix,y):
	return 1-np.dot(np.asarray(ix).T,y)/(np.linalg.norm(np.asarray(ix))*np.linalg.norm(y))
def test(inputs, targets):
  xs, hs, ys = {}, {}, {}
  hs[-1] = np.zeros((hidden_size,1))
  # forward pass
  for t in range(len(inputs)):
    a = np.reshape(inputs[t],(300,1))
    xs[t] = np.zeros((vocab_size_en,1)) 
    xs[t] = np.copy(a)
    hs[t] = np.tanh(np.dot(wxh_en, xs[t]) + np.dot(whh_en, hs[t-1]) + bh_en) # hidden state
    ys[t] = np.dot(why_en, hs[t]) + by_en 
  hprev = hs[len(inputs)-1]
  tem = ""
  ans = ""
  k = 0
  t = -1
  print('Encoder done -- Finding the closest vector and decoding')
  while k<=len(targets):
      x = np.zeros((vocab_size_fr,1))
      if(t!=-1):
          x[t] = 1
      hprev = np.tanh(np.dot(wxh_fr, x) + np.dot(whh_fr, hprev) + bh_fr)
      y = np.dot(why_fr, hprev) + by_fr # unnormalized log probabilities for next chars
      keys = np.zeros((len(ix_to_word_fr)))
      mind = 100000.0
      min_key = tuple
      for ix in ix_to_word_fr.keys():
          cosdist = (cosine_dist(ix,y))
          if(cosdist<mind):
                print(cosdist,ix_to_word_fr[ix],ix_to_word_en[tuple(targets[k-1])])
                mind = cosdist
                minkey = ix
      tem = ix_to_word_fr[minkey]
      k = k + 1
      ans = ans + " " +tem
  return ans

  
def testcase(sentence):
	print('-----------------------------------------------------')
	input_english = sentence
	print('Testing sentence : \'%s\''%(input_english),end='\n\n')
	curr_en = input_english.split()
	inputs_en = [word_to_ix_en[w] for w in curr_en[0:len(curr_en)-1]]
	targets_en = [word_to_ix_en[w] for w in curr_en[1:len(curr_en)]]
	output_words = test(inputs_en,targets_en)
	print('-----------------------------------------------------')
	print('[EN]',sentence,'=>',output_words,'[FR]')
	print('-----------------------------------------------------')
	print(cosine_dist(model_fr.wv['United'],model_fr.wv[output_words.split()[0]]))

testcase("In the opinion of the")
while(True):
	sentence = input("Enter Sentence to translate [EN]->[FR]\n")
	if(sentence == '' or sentence == '\n'):
		close()
	else:
		testcase(sentence)
		
#Revised provisions - Résumé additionnelles,
#United Nations Secretariat 
#In the opinion of the =>  viennent combattants la Croatie souffrance; 
#In the opinion of the committee =>  viennent combattants la Croatie souffrance; autrichien
#Fifty-fourth session =>  confier additionnelles,
#The Council further decided =>  l'alimenter. culture la Croatie