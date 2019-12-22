import numpy as np
import visen

CHAR_LIST = "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
VOWEL_LIST = list("aeiouàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
CONSONANT_LIST = list("bcdfghjklmnpqrstvwxyz")

def delete_char(word, num_char=1):
    word = [c for c in word]
    n = len(word)
    num_char = min(num_char, n)
    pos = np.random.choice(range(1,n), num_char, replace=False)
    for i in pos:
        word[i] = ''
    word = ''.join(word)
    return word

def substitute_char(word, num_char=1):
    word = [c for c in word]
    n = len(word)
    num_char = min(num_char, n)
    pos = np.random.choice(range(1,n), num_char, replace=False)
    for i in pos:
        c = word[i]
        if c in VOWEL_LIST:
            c = np.random.choice(VOWEL_LIST)
        else:
            c = np.random.choice(CONSONANT_LIST)
        word[i] = c
    word = ''.join(word)
    return word


def transform_sentence(sent, p_transform=0.4, p_del=0.1, p_sub=0.9, p_tf_word={'del':0.5, 'sub':0.5}):
    words = sent.split()
    tf_type = np.random.choice([0,1,2], len(words), p=[1-p_transform, p_transform*p_del, p_transform*p_sub])
    for i,t in enumerate(tf_type):
        if t==1:
            words[i] = ''
        elif t==2:
            n_iter = np.random.randint(1,3)
            funcs = np.random.choice([delete_char, substitute_char], n_iter, p=[p_tf_word['del'], p_tf_word['sub']])
            for func in funcs:
                if len(words[i]) < 2:
                    break
                words[i] = func(words[i])
    words = [w for w in words if w!='']
    return ' '.join(words)

