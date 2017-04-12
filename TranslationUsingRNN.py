import numpy as np

lines_en = open('test.en','r',encoding='utf8').readlines()
data_en = open('test.en','r',encoding='utf8').read()
data_en = data_en.split()
words_en = list(set(data_en))
data_size_en,vocab_size_en = len(data_en),len(words_en)
print('english file has %d words,%d unique'%(data_size_en,vocab_size_en))
word_to_ix_en = { w:i for i,w in enumerate(words_en)}
ix_to_word_en = { i:w for i,w in enumerate(words_en)}

print("Word to ix:",word_to_ix_en))
print("ix to Word:", ix_to_word_en[0:100])
#taking the french data and creating mappings for one hot encoding
lines_fr = open('test.fr','r',encoding='utf8').readlines()
data_fr = open('test.fr','r',encoding='utf8').read()
data_fr = data_fr.split()
words_fr = list(set(data_fr))
data_size_fr,vocab_size_fr = len(data_fr),len(words_fr)
print('french file has %d words,%d unique'%(data_size_fr,vocab_size_fr))
word_to_ix_fr = { w:i for i,w in enumerate(words_fr)}
ix_to_word_fr = { i:w for i,w in enumerate(words_fr)}

#Number of sentences in training data
num = len(lines_en)
num = len(lines_fr)
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

def trainencoder(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_en,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_en, xs[t]) + np.dot(whh_en, hs[t-1]) + bh_en) # hidden state
    ys[t] = np.dot(why_en, hs[t]) + by_en # unnormalized log probabilities for next words
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next words

  dWxh, dWhh, dWhy = np.zeros_like(wxh_en), np.zeros_like(whh_en), np.zeros_like(why_en)
  dbh, dby = np.zeros_like(bh_en), np.zeros_like(by_en)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(why_en.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(whh_en.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def traindecoder(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_fr,1)) # encode in 1-of-k representation
    if(inputs[t]!=-1):
        xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_fr, xs[t]) + np.dot(whh_fr, hs[t-1]) + bh_fr) # hidden state
    ys[t] = np.dot(why_fr, hs[t]) + by_fr # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

  dWxh, dWhh, dWhy = np.zeros_like(wxh_fr), np.zeros_like(whh_fr), np.zeros_like(why_fr)
  dbh, dby = np.zeros_like(bh_fr), np.zeros_like(by_fr)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(why_fr.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(whh_fr.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def test(inputs, targets):
  xs, hs, ys = {}, {}, {}
  hs[-1] = np.zeros((hidden_size,1))
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size_en,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(wxh_en, xs[t]) + np.dot(whh_en, hs[t-1]) + bh_en) # hidden state
    ys[t] = np.dot(why_en, hs[t]) + by_en 
  hprev = hs[len(inputs)-1]
  tem = ""
  ans = ""
  k = 0
  t = -1
  while k<10:
      x = np.zeros((vocab_size_fr,1))
      if(t!=-1):
          x[t] = 1
      hprev = np.tanh(np.dot(wxh_fr, x) + np.dot(whh_fr, hprev) + bh_fr)
      y = np.dot(why_fr, hprev) + by_fr # unnormalized log probabilities for next chars
      pr = np.exp(y) / np.sum(np.exp(y))
      maxi = pr[0][0]
      for i in range(len(words_fr)):
          if pr[i][0]>=maxi:
              maxi = pr[i][0]
              t = i
      tem = ix_to_word_fr[t]
      k = k + 1
      ans = ans + " " +tem
  return ans;
    
n,p = 1,0
mwxh_en,mwhh_en,mwhy_en,mbh_en,mby_en = np.zeros_like(wxh_en),np.zeros_like(whh_en),np.zeros_like(why_en),np.zeros_like(bh_en),np.zeros_like(by_en)
mwxh_fr,mwhh_fr,mwhy_fr,mbh_fr,mby_fr = np.zeros_like(wxh_fr),np.zeros_like(whh_fr),np.zeros_like(why_fr),np.zeros_like(bh_fr),np.zeros_like(by_fr)


while n!=2000:
    curr_en = lines_en[p].split()
    inputs_en = [word_to_ix_en[w] for w in curr_en[0:len(curr_en)-1]]
    targets_en = [word_to_ix_en[ch] for ch in curr_en[1:len(curr_en)]]
    
    curr_fr = lines_fr[p].split()
    inputs_fr=[-1]
    temp = [word_to_ix_fr[ch] for ch in curr_fr[0:len(curr_fr)-1]]
    inputs_fr.extend(temp)
    targets_fr = [word_to_ix_fr[ch] for ch in curr_fr[0:len(curr_fr)]]

    hprev = np.zeros((hidden_size,1))
    dwxh_en,dwhh_en,dwhy_en,dbh_en,dby_en,hprev = trainencoder(inputs_en,targets_en,hprev)
    dwxh_fr,dwhh_fr,dwhy_fr,dbh_fr,dby_fr,hprev = traindecoder(inputs_fr,targets_fr,hprev)
    
    p += 1 # move sentence pointer
    if p >= num:
        p = 0
        print('training...iteration:%d'%(n))
        #input_english = input('english:')
        input_english = "A note on the outcome of the discussions at the retreat will be presented separately."
        curr_en = input_english.split()
        inputs_en = [word_to_ix_en[ch] for ch in curr_en[0:len(curr_en)-1]]
        targets_en = [word_to_ix_en[ch] for ch in curr_en[1:len(curr_en)]]
        output_words = test(inputs_en,targets_en)
        print(output_words)  
        n = n + 1
    for param_en, dparam_en, mem_en in zip([wxh_en, whh_en, why_en, bh_en, by_en], 
                                [dwxh_en, dwhh_en, dwhy_en, dbh_en, dby_en], 
                                [mwxh_en, mwhh_en, mwhy_en, mbh_en, mby_en]):
      mem_en += dparam_en * dparam_en
      param_en += -learning_rate * dparam_en / np.sqrt(mem_en + 1e-8) # adagrad update
    





    for param_fr, dparam_fr, mem_fr in zip([wxh_fr, whh_fr, why_fr, bh_fr, by_fr], 
                                [dwxh_fr, dwhh_fr, dwhy_fr, dbh_fr, dby_fr], 
                                [mwxh_fr, mwhh_fr, mwhy_fr, mbh_fr, mby_fr]):
      mem_fr += dparam_fr * dparam_fr
      param_fr += -learning_rate * dparam_fr / np.sqrt(mem_fr + 1e-8) # adagrad update
 # iteration counter 



                 
    

