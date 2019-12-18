def create_counter_dict(words):
    counter_dict = dict()
    for word in words:
        increase_counter(counter_dict, word)
    return counter_dict


def increase_counter(words_counter_dict, word):
    words_counter_dict[word] = words_counter_dict.get(word, 0) + 1


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


def get_words_list(filename):
    words_list = list()
    lines = read_lines(filename)
    for i in range(2, len(lines), 4):
        for word in lines[i].split():
            words_list.append(word)

    return words_list
