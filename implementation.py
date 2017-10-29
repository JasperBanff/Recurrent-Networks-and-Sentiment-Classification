import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50


stopwords = ['A', 'After', 'All', 'American', 'And', 'As', 'At', 'British', 'But', 'DVD', 'Even', 'For', 'He', 'His', 'Hollywood', 'How', 'However,', 'I', "I'd", "I'm", "I've", 'If', 'In', 'It', "It's", 'John', 'Just', 'Michael', 'Mr.', 'My', 'New', 'Of', 'On', 'One', 'She', 'So', 'Some', 'THE', 'TV', 'That', 'The', 'There', "There's", 'They', 'This', 'To', 'We', 'What', 'When', 'While', 'With', 'You', 'a', 'able', 'about', 'absolutely', 'across', 'act', 'acting', 'action', 'actor', 'actors', 'actress', 'actual', 'actually', 'after', 'again', 'again,', 'again.', 'against', 'all', 'all,', 'all.', 'almost', 'along', 'already', 'also', 'although', 'always', 'am', 'an', 'and', 'and,', 'another', 'any', 'anyone', 'anything', 'appears', 'are', 'around', 'as', 'at', 'audience', 'away', 'back', 'based', 'be', 'because', 'become', 'becomes', 'been', 'before', 'beginning', 'behind', 'being', 'believe', 'between', 'big', 'bit', 'black', 'book', 'both', 'boy', 'bring', 'budget', 'but', 'by', 'called', 'came', 'camera', 'car', 'care', 'case', 'cast', 'certainly', 'character', 'characters', 'characters,', 'child', 'children', 'classic', 'clearly', 'close', 'come', 'comedy', 'comes', 'coming', 'complete', 'completely', 'couple', 'course', 'course,', 'day', 'dead', 'death', 'definitely', 'despite', 'dialogue', 'did', 'different', 'directed', 'direction', 'director', 'do', 'does', 'doing', 'done', 'down', 'due', 'during', 'each', 'early', 'easily', 'effects', 'either', 'else', 'end', 'end,', 'end.', 'ending', 'ends', 'enough', 'entire', 'episode', 'especially', 'even', 'ever', 'every', 'everyone', 'everything', 'evil', 'exactly', 'example', 'expect', 'extremely', 'eyes', 'face', 'fact', 'fact,', 'fall', 'falls', 'family', 'fan', 'fans', 'far', 'father', 'feel', 'feeling', 'feels', 'felt', 'female', 'few', 'fight', 'film', "film's", 'film,', 'film.', 'films', 'films,', 'final', 'finally', 'find', 'finds', 'fine', 'first', 'for', 'found', 'four', 'friend', 'friends', 'from', 'full', 'fun', 'funny', 'game', 'gave', 'get', 'gets', 'getting', 'girl', 'girls', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'good', 'good,', 'good.', 'got', 'great', 'group', 'guess', 'guy', 'guys', 'had', 'half', 'happened', 'happens', 'hard', 'has', 'have', "haven't", 'having', 'he', "he's", 'head', 'hear', 'heard', 'help', 'her', 'here', 'here,', 'here.', 'high', 'him', 'him.', 'himself', 'his', 'hit', 'home', 'hope', 'horror', 'house', 'how', 'however,', 'huge', 'human', 'humor', 'i', 'idea', 'if', 'in', 'instead', 'into', 'involved', 'is', 'is,', "isn't", 'it', "it's", 'it,', 'it.', 'it.<br', 'its', 'itself', 'job', 'just', 'keep', 'kids', 'kill', 'kind', 'knew', 'know', 'known', 'last', 'late', 'later', 'laugh', 'lead', 'least', 'leave', 'left', 'less', 'let', 'life', 'life.', 'line', 'lines', 'little', 'live', 'lives', 'local', 'long', 'look', 'looking', 'looks', 'lost', 'lot', 'low', 'made', 'main', 'major', 'make', 'makes', 'making', 'man', 'many', 'matter', 'may', 'maybe', 'me', 'me,', 'me.', 'mean', 'men', 'might', 'mind', 'minutes', 'moment', 'moments', 'money', 'more', 'most', 'mostly', 'mother', 'movie', 'movie,', 'movie.', 'movies', 'movies,', 'movies.', 'much', 'music', 'must', 'my', 'myself', 'name', 'nearly', 'need', 'needs', 'never', 'new', 'next', 'nice', 'night', 'no', 'not', 'nothing', 'now', 'number', 'obvious', 'of', 'off', 'often', 'old', 'on', 'once', 'one', 'one,', 'one.', 'only', 'opening', 'or', 'order', 'original', 'other', 'others', 'our', 'out', 'out.', 'over', 'own', 'part', 'particularly', 'parts', 'past', 'people', 'performance', 'performances', 'perhaps', 'person', 'picture', 'piece', 'place', 'play', 'played', 'playing', 'plays', 'plot', 'point', 'police', 'pretty', 'probably', 'problem', 'production', 'put', 'quality', 'quite', 'rather', 'read', 'real', 'really', 'reason', 'remember', 'rest', 'role', 'run', 'said', 'same', 'saw', 'say', 'says', 'scene', 'scenes', 'school', 'screen', 'script', 'second', 'see', 'seeing', 'seem', 'seemed', 'seems', 'seen', 'sense', 'series', 'serious', 'set', 'several', 'sex', 'she', "she's", 'short', 'shot', 'should', 'show', 'shown', 'shows', 'side', 'simply', 'since', 'small', 'so', 'some', 'someone', 'something', 'somewhat', 'soon', 'sort', 'sound', 'star', 'stars', 'start', 'started', 'starts', 'still', 'stop', 'stories', 'story', 'story,', 'story.', 'strange', 'style', 'such', 'sure', 'take', 'taken', 'takes', 'taking', 'tell', 'than', 'that', "that's", 'that,', 'that.', 'the', 'their', 'them', 'them.', 'themselves', 'then', 'there', "there's", 'these', 'they', "they're", 'thing', 'things', 'think', 'thinking', 'this', 'this,', 'this.', 'those', 'though', 'thought', 'three', 'through', 'throughout', 'time', 'time,', 'time.', 'times', 'title', 'to', 'together', 'told', 'too', 'took', 'top', 'totally', 'town', 'tries', 'true', 'truly', 'try', 'trying', 'turn', 'turned', 'turns', 'two', 'type', 'under', 'understand', 'until', 'up', 'upon', 'us', 'use', 'used', 'usual', 'usually', 'version', 'very', 'video', 'viewer', 'voice', 'want', 'wanted', 'wants', 'was', "wasn't", 'watch', 'watched', 'watching', 'way', 'way,', 'way.', 'we', 'well', 'well,', 'well.', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'white', 'who', 'whole', 'whose', 'why', 'wife', 'will', 'wish', 'with', 'without', 'woman', 'women', "won't", 'work', 'world', 'worth', 'would', "wouldn't", 'writing', 'written', 'wrong', 'year', 'years', 'yet', 'you', "you'll", "you're", 'young', 'your']
def check_file(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
    return filename

def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'reviews/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'reviews/'))
    return

def read_data(glove_dict, seq_length):
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'reviews/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'reviews/neg/*')))
    for f in file_list:
        vector = []
        with open(f, "r", encoding = "utf-8") as openf:
            words = openf.read().split()
            for word in words:
                if len(word) <=2 or word in stopwords:
                    continue
                elif word[0].isdigit() or not word[0].isalpha():
                    continue
                elif not word[-1].isalpha():
                    word = word[:-1]
                    if word in glove_dict:
                        vector.append(glove_dict[word])
                    else:
                        continue
                else:
                    if word in glove_dict:
                        vector.append(glove_dict[word])
                    else:
                        continue
                if len(vector) == seq_length:
                    break
            residue = seq_length - len(vector)
            for _ in range(residue):
                vector.append(0)
            data.append(vector)
    data = np.array(data)
    return data


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    seq_length = 40  # Maximum length of sentence
    data = []
    filename = check_file('reviews.tar.gz', 14839260)
    extract_data(filename) # unzip
    data = read_data(glove_dict, seq_length)   

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    embeddings = list()
    word_index_dict = dict()
    idx = 0
    for words in data:
        wordList = words.split()
        word_index_dict[wordList[0]] = idx
        word_embedding_vector = list()
        for word in wordList[1:]:
            word_embedding_vector.append(np.float32(word)) 
        embeddings.append(word_embedding_vector)
        idx += 1
    
    word_index_dict['UNK'] = idx
    embeddings.append([np.float32(0)]*len(embeddings[0]))
    #transfer the list to array
    embeddings = np.asarray(embeddings)
    return embeddings, word_index_dict


def BiRNN(x,dropout_keep_prob):
    with tf.variable_scope('bi_directional_lstm') as scope:
        lstmUnits = 64
          

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=dropout_keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=dropout_keep_prob)
        #outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell, x, dtype = tf.float32)
        
        (value_fw,value_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw =lstm_fw_cell, cell_bw =  lstm_bw_cell, dtype = tf.float32, inputs = x)

        value = tf.concat((value_fw, value_bw),2)
        last_output = value[:,-1,:]

        prediction = tf.layers.dense(last_output,2)

    return prediction



def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    # glove_embeddings_arr  shape = (vocabulary_size, 50) 
    #Number of LSTM units: 
    #This value is largely dependent on the average length of your input texts.
    #While a greater number of units provides more expressibility
    #for the model and allows the model to store more information
    #for longer texts, the network will take longer to 
    #train and will be computationally expensive.

    

    numDimensions = 50
    maxSeqLength = 40
    lstmUnits = 64
    numClasses = 2
    #num_layers = 2

    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, shape = [None,numClasses], name = 'labels')
    input_data = tf.placeholder(tf.int32,shape = [None,maxSeqLength], name = 'input_data')
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape = ())

    #data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]), dtype = tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr,input_data)


    #weights = tf.Variable(tf.truncated_normal([2*lstmUnits, numClasses]))
    #biases = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    

    lstm_fw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
    lstm_bw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=dropout_keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    #outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell, x, dtype = tf.float32)
        
    (value_fw,value_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw =lstm_fw_cell, cell_bw =  lstm_bw_cell, dtype = tf.float32, inputs = data)

    value = tf.concat((value_fw, value_bw),2)
    last_output = value[:,-1,:]

    prediction = tf.layers.dense(last_output,2)

    # see the accuracy
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name = 'accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name = 'loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels,dropout_keep_prob, optimizer, accuracy, loss


