import random
import numpy as np
import seaborn as sns


class Dicegame:
    def __init__(self)-> None:
        pass

    def play_game(self, play=True):


    def play_dice(self, play=True):
        if not play:
            return 10
        res = 0
        playing= True
        while playing:
            random.randint(1, 6)
            if rdn > 2:
                res+= 4
            else:
                res+= 4
                return  res