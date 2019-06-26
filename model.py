#importing all libraries
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.squad import data
from tensorlayer.models.seq2seq import Seq2seq
from seq2seq_attention import Seq2seqLuongAttention
import os
import sqlite3
import spacy
from textblob import TextBlob
import time
import sys
import pickle

print('All libraries imported')


def load_data(PATH=''):
    # read data control dictionaries
    try:
        with open(PATH + 'metadata_q.pkl', 'rb') as f1:
            metadata_q = pickle.load(f1)
        with open(PATH + 'metadata_a.pkl', 'rb') as f2:
            metadata_a = pickle.load(f2)

    except:
        metadata_q = None
        metadata_a = None
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata_q, metadata_a, idx_q, idx_a


def initial_setup(data_corpus):
    metadata_q, metadata_a, idx_q, idx_a = load_data(PATH='data/{}/'.format(data_corpus)) 
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata_q, metadata_a, trainX, trainY, testX, testY, validX, validY



if __name__ == "__main__":
    data_corpus = "squad"

    #data preprocessing
    metadata_q, metadata_a, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)
    
    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    print(src_len,tgt_len)
    
    #sys.exit(0)

    assert src_len == tgt_len

    batch_size = 32
    n_step = src_len // batch_size
    src_vocab_size = len(metadata_q['idx2w'])
    
    #tgt_vocab_size = len(metadata_a['idx2w']) # 8002 (0~8001)
    #print(src_vocab_size,tgt_vocab_size)
    #sys.exit(0)
    
    
    
    emb_dim = 1024

    word2idx = metadata_a['w2idx']   # dict  word 2 index
    word2idx_2 = metadata_q['w2idx']
    word2idx.update(word2idx_2)

    idx2word= metadata_a['idx2w']   # list index 2 word
    idx2word_2 = metadata_q['idx2w']
    idx2word.append(idx2word_2)
    #print(type(idx2word))
    #print(word2idx)    
    #sys.exit(0)
   
    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    num_epochs = 1
    vocabulary_size = src_vocab_size
    
    count=0 #For keeping count of entries into db
    
    print('data processing done')

    def inference(seed, top_n):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    decoder_seq_length = 20
    '''
    model_=model_ = Seq2seqLuongAttention(
            hidden_size=128, cell=tf.keras.layers.SimpleRNNCell,
            embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size,
                                                embedding_size=emb_dim), method='dot'
)
    '''


    model_ = Seq2seq(
        decoder_seq_length = decoder_seq_length,
       cell_enc=tf.keras.layers.LSTMCell,
      cell_dec=tf.keras.layers.LSTMCell,
     n_layer=3,
     n_units=256,
     embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim))
    

    # Uncomment below statements if you have already saved the model

    # load_weights = tl.files.load_npz(name='model.npz')
    # tl.files.assign_weights(load_weights, model_)
    load_weights = tl.files.load_hdf5_to_weights('model.hdf5', model_, skip=False)
    tl.files.assign_weights(load_weights, model_)

    print('loaded model')

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model_.train()



    db=sqlite3.connect('chatbot.db')
    cursor=db.cursor()
    cursor.execute('''create table user_inputs(questions TEXT)''')
    db.commit()

    #seeds = ["happy birthday have a nice day",
               #  "donald trump won last nights presidential debate according to snap online polls"]
    for epoch in range(num_epochs):
        model_.train()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, _decode_seqs])
                
                output = tf.reshape(output, [-1, vocabulary_size])
                ## compute loss and update model
                loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                grad = tape.gradient(loss, model_.all_weights)
                optimizer.apply_gradients(zip(grad, model_.all_weights))
            
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
        tl.files.save_weights_to_hdf5('model.hdf5', model_)
        print("model saved") 
        
    #tl.files.save_weights_to_hdf5('model.hdf5', model_)
    #print("model saved")   
        
    i=0
    while(i<10):
        user_input=input()
        if(user_input=="Bye"):
            print("Bye")
            i=11
            break
        else:
            blob=TextBlob(user_input)
            if(blob.sentiment.polarity<0):
                count+=1
                cursor.execute('''insert into user_inputs(questions) VALUES(?)''', (user_input,))
                db.commit()
            else:
                pass
                
            print("Query >", user_input)
            top_n=3
            for i in range(top_n):
                sentence=inference(user_input, top_n)
                print(">",' '.join(sentence))
    
    
           

#Follow up
#create db, input respones to db
#if db is negative, store it send it back to model afte some time decided by the decision system
   
    

        #insert into table
    db.close()

    #New start
    #follow up
    #Loading model
    load_weights = tl.files.load_hdf5_to_weights('model.hdf5', model_, skip=False)
    tl.files.assign_weights(load_weights, model_)
      
    #Really need to do the decision system
    time.sleep(5)

    print('Beginning of follow up')
    #Connect to db and take in inputs
    db=sqlite3.connect('chatbot.db')
    cursor=db.cursor()
    inputs=cursor.fetchall()
    
    seeds=[]
    for i in range(count):
        a=list(inputs[i])
        a=''.join(a)
        seeds.append(a)

    for seed in seeds:
            print("Query >", seed)
            top_n = 3
            for i in range(top_n):
                sentence = inference(seed, top_n)
                print(" >", ' '.join(sentence))
