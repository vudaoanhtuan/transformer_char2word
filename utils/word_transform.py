import numpy as np
import visen

CHAR_LIST = "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"

def delete_char(word, num_char=1):
    word = [c for c in word]
    n = len(word)
    if n>2:
        pos = np.random.choice(range(1,n), num_char, replace=False)
        for i in pos:
            word[i] = ''
    word = ''.join(word)
    return word


def short_word(word):
    sw_dict = {
        'không': ['k', 'ko', 'k0', 'kg', 'hk'],
        'vậy': ['z', 'zậy'],
        'gì': ['j', 'gj'],
    }

def get_teen_code(word):
    def vz(word):
        # vậy -> zậy
        word = word.replace('v', 'z')
        return word
    
    def ij(word):
        # vui -> vuj
        word = word.replace('i', 'j')
        return word

    def qw(word):
        # qua -> wua
        word = word.replace('q', 'w')
        return word

    def quw(word):
        # qua -> wa
        word = word.replace('qu', 'w')
        return word
    
    def ng(word):
        # không -> khôg
        if word.endswith('ng'):
            word = word[:-2] + 'g'
        return word
    
    for func in [vz, ij, quw, qw, ng]:
        word = func(word)
    
    return word

def remove_tone(word):
    word = visen.remove_tone(word)
    return word

def get_enter_code(word):
    code = visen.format.get_enter_code(word)
    return code

def get_typo(word):
    word = get_enter_code(word)
    num_del = [1,2]
    pc = [0.85, 0.15]
    if len(word) < 5:
        num_del = [1,1]
    n = np.random.choice(num_del, p=pc)
    word = delete_char(word, n)
    return word

def gen_wrong_word(word):
    func = [typo, teen_code, remove_tone]
    func = np.random.choice(func)
    word = func(word)
    return word
