def get_kth_character(k):
    word = "a"
    while len(word) < k:
        new_word = ''.join(chr((ord(c) - ord('a') + 1) % 26 + ord('a')) for c in word)
        word = word + new_word
    return word[k - 1]



def get_kth_character(k):
    if k == 1:
        return 'a'
    length = 1
    while length * 2 < k:
        length *= 2
    return chr((ord(get_kth_character(k - length)) - ord('a') + 1) % 26 + ord('a'))
