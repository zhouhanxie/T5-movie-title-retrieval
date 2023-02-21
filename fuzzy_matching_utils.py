import re
import regex
from copy import deepcopy

class Mention:
    
    def __init__(self, surface, root, root_pos = None, to_string_option='info'):
        self.surface = surface
        self.root = root
        self.root_pos = root_pos
        self.to_string_option = to_string_option
        
    def info(self):
        return 'Mention(surface:'+self.surface+', root: '+self.root+', pos: '+str(self.root_pos)+')'
    
    def __str__(self):
        if self.to_string_option=='info':
            return self.info()
        return self.surface
    
    def __repr__(self):
        return str(self)
    
    def set_pos(self, val):
        self.root_pos = val
        return self

def filter_set(candidates, text):
    """
    given a list of candidate strings and a text,
    remove all candidates that does not have exact match.
    for the remaining, for any pair of string a, b, always keep
        b only if a is substring of b
    """
    filtered = [c for c in candidates if c in text]
    filtered = sorted(filtered, key=lambda x:len(x), reverse=True)
    out = []
    while len(filtered) > 0:
        cur = filtered.pop()
        cur_useful = True
        for other in filtered:
            if cur in other:
                cur_useful = False 
        if cur_useful:
            out.append(cur)
    return set(out)


def match_all(needle, haystack):
        
    out = [deepcopy(needle).set_pos(m.span()[0]) for m in re.finditer(needle.surface, haystack)]
    return out

def get_surface_mention(needle, haystack, edit_threshold=2):
    """
    Return all fuzzy matched keywords, within edit distanec.
    Empty list returned if no match.
    """
    rule = regex.compile('(?b)(' + needle + '){s<='+str(edit_threshold)+'}') 
    return [Mention(surface, needle, to_string_option='info') for surface in regex.findall(rule, haystack)]


def fuzzy_findall(needles, haystack, edit_threshold=2, sort=True):
    """
    needles: list of keywords
    haystack: your string document
    """
    out = []
    for needle in needles:
        mentions = get_surface_mention(needle, haystack, edit_threshold=edit_threshold)
        for mention in mentions:
            out += match_all(mention, haystack)
    out = sorted(out, key=lambda x:x.root_pos)
    return out