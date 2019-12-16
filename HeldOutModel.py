from utils import create_counter_dict


class HeldOutModel:
    def __init__(self, words_list, X, split_factor=0.5):
        if split_factor > 1 or split_factor < 0:
            raise Exception('Invalid split factor')
        self.__total_size = len(words_list)
        self.__training_size = round(self.__total_size * split_factor)
        self.__held_out_size = self.__total_size - self.__training_size
        self.__training = create_counter_dict(words_list[:self.__training_size])
        self.__held_out = create_counter_dict(words_list[self.__training_size:])
        self.__N0 = X - len(self.__training.keys())
        self.__unseen_events = [event for event in self.__held_out.keys() if event not in self.__training.keys()]
        self.__create_prob_dict()

    def get_training_size(self):
        return self.__training_size

    def get_held_out_size(self):
        return self.__held_out_size

    def __create_prob_dict(self):
        self.prob_dict = dict()
        for r in range(0, max(self.__training.values()) + 1):
            self.__calc_prob(r)

    def get_prob(self, event):
        return self.prob_dict[self.__training.get(event, 0)]

    def __calc_prob(self, r):
        if r == 0:
            r_events = self.__unseen_events
            Nr = self.__N0
        else:
            r_events = [event for event, count in self.__training.items() if count == r]
            if len(r_events) == 0:
                return
            Nr = len(r_events)

        tr = sum([count for event, count in self.__held_out.items() if event in r_events])

        self.prob_dict[r] = tr / (self.__held_out_size * Nr)

    def debug(self):
        sum_p = self.get_prob('unseen-word') * self.__N0
        for event in self.__training.keys():
            sum_p += self.get_prob(event)
        return sum_p
