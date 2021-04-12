from pathlib import Path
import datetime
import random

import inspect
import neuroseg.utils as utils
utils_path = Path(inspect.getfile(utils)).parent

animals_path = utils_path.joinpath("animals.txt")
adjective_path = utils_path.joinpath("adjectives.txt")


class NameGenerator:
    
    def __init__(self,
                 noun_txt_path=animals_path,
                 adjective_txt_path=adjective_path,
                 include_time=True):
        
        self.noun_txt_path = Path(noun_txt_path)
        self.adjective_txt_path = Path(adjective_txt_path)
        
        self.noun_list = self._read_file(self.noun_txt_path)
        self.adjective_list = self._read_file(self.adjective_txt_path)
        
        self.current_datetime = datetime.datetime.now()
        self.include_time = include_time
        
        self._parse_date()
        self._gen_name()

        
    @staticmethod
    def _read_file(filepath):
        return filepath.read_text().splitlines()
    
    def _parse_date(self):
        self.current_datetime = datetime.datetime.now()
        self.year = self.current_datetime.year
        self.month = self.current_datetime.month
        self.day = self.current_datetime.day
        
        self.datestring = ("-{}{}").format(self.month, self.day)
        
    def _gen_name(self):
        
        self.noun = random.choice(self.noun_list)
        self.adjective = random.choice(self.adjective_list)
        
        self.name = "{}-{}".format(self.adjective, self.noun)
        if self.include_time:
            self.name = self.name+self.datestring