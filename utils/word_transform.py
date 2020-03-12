import numpy as np
import visen

CHAR_LIST = list("abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
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


def insert_char(word):
    i = np.random.randint(len(word)+1)
    t1 = word[i%len(word)] in VOWEL_LIST
    t2 = word[i-1] in VOWEL_LIST
    if t1 and t2:
        c = np.random.choice(VOWEL_LIST)
    elif not (t1 and t2):
        c =  np.random.choice(CONSONANT_LIST)
    else:
        c = np.random.choice(CHAR_LIST)
    word = word[:i] + c + word[i:]
    return word


def remove_tone(word):
    word = visen.remove_tone(word)
    return word

def get_enter_code(word):
    code = visen.format.get_enter_code(word)
    return code

def get_typo(word):
    word = get_enter_code(word)
    if len(word) < 2:
        return word
    num_del = [1,2]
    pc = [0.85, 0.15]
    if len(word) < 5:
        num_del = [1,1]
    n = np.random.choice(num_del, p=pc)
    word = delete_char(word, n)
    return word

def transform_sentence(sent, p_transform=0.6, p_del=0.3, p_ins=0.3, p_sub=0.4):
    chance = np.random.choice([0,1,2], p=[0.05, 0.2, 0.75])
    if chance == 1:
        sent = visen.remove_tone(sent)
        return sent
    elif chance == 2:
        words = sent.split()
        p_transform = np.random.randint(10,int(p_transform*100)) * 0.01
        tf_type = np.random.choice([0,1], len(words), 
            p=[1-p_transform, p_transform])
        for i,t in enumerate(tf_type):
            if t==1:
                is_rm_tone = np.random.choice([True, False], p=[0.4, 0.6])
                if is_rm_tone:
                    words[i] = remove_tone(words[i])
                else:
                    n_iter = np.random.choice([1,2,3], p=[0.8, 0.18, 0.02])
                    funcs = np.random.choice([delete_char, insert_char, substitute_char], n_iter, 
                        p=[p_del, p_ins, p_sub])
                    for func in funcs:
                        if len(words[i]) < 2:
                            break
                        words[i] = func(words[i])
        words = [w for w in words if w!='']
        return ' '.join(words)
    return sent