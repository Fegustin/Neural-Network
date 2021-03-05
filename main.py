from AI.CinF import translate
from AI.DefinitionOfNumbers import definition
from AI.BostonHousen import price
from AI.Tunner import run_tun
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # translate()
    definition()
    # price()
    # run_tun()
