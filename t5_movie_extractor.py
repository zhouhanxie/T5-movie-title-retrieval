import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from spacy.lang.en import English

# useful for sentencizing
nlp = English()
nlp.add_pipe('sentencizer')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    """https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sentencize(sentence):
    """
    input:
        string or a list of string
    output:
        list of string, where each input document
        is split into sentences using sentencizer
        
        another chunk identifier List[int]
        remembers which group of sentences is corresponding to the original
        text
    """
    if type(sentence) == str:
        docs = [sentence]
    else:
        docs = sentence
    outputs = []
    chunk_identifiers = []
    for d in docs:
        sentences_this_d = [sent.text.strip() for sent in nlp(d).sents]
        outputs += sentences_this_d
        # chunk_identifiers.append(d)
        chunk_identifiers.append(len(sentences_this_d))
        
    return outputs, chunk_identifiers

def desentencize(items, chunk_identifiers, group_strategy=lambda x:' '.join(x)):
    output = []
    ptr = 0
    for chunk_size in chunk_identifiers:
        this_chunk = items[ptr:ptr+chunk_size]
        output.append(group_strategy(this_chunk))
        ptr += chunk_size
    return output

class T5MovieExtractor:
    
    def __init__(
        self, 
        model_name_or_path,
        device='cuda:0', 
        max_length=148, 
        prompt = 'Perform named entity recognition to extract movie titles. '\
        'List all movie titles that appears in text above. '\
        'Separate movie names using comma, and do not include years, genre, director, award '\
        'and other movie-related word that are not titles.',
        entity_dictionary = None,
        do_sentencize = True
    ):
        if entity_dictionary is not None:
            raise ValueError('T5 Movie Extractor currently does not support custom entity for fuzzy matching')
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)
        self.prompt = prompt
        self.do_sentencize = do_sentencize
        self.chunk_identifiers = None
        
    def _extract(self, raw_inp):

        inp = [self.prompt+i.replace('\n','') for i in raw_inp]
            
        with torch.no_grad():
            encoded_src = self.tokenizer(
                inp, 
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            output = self.model.generate(**encoded_src, max_length=128)
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        
        return output
    
    def __call__(self, sentence, batch_size=4, disable_tqdm=False, on_error='raise'):
        """
        sentence: Union[List, String]
        batch_size: int
        distable_tqdm: bool
        on_error: string in {'raise', 'echo', 'nothing'}
        """
        assert on_error in {'raise', 'echo', 'nothing'}
        if type(sentence) not in (list, str):
            raise TypeError
        if type(sentence) == str:
            raw_inp = [sentence]
        else:
            raw_inp = sentence

        if self.do_sentencize:
            raw_inp, identifiers = sentencize(raw_inp)
            self.chunk_identifiers = identifiers 
            
        batches = list(chunks(raw_inp, batch_size))
        outputs = []
        
        for batch in tqdm(batches, disable=disable_tqdm):
            try:
                outputs += self._extract(batch)
            except KeyboardInterrupt:
                raise
            except:
                if on_error == 'nothing':
                    outputs.append('none')
                elif on_error == 'echo':
                    print('---')
                    print('T5MovieExtractor: oops broken batch, sorry...')
                    print(batch)
                    print('---')
                    outputs.append('none')
                raise ValueError('the batch above broke the pipeline')
            
        if self.do_sentencize:
            outputs = desentencize(outputs, self.chunk_identifiers, lambda x:','.join(x))

        if type(sentence) is str:
            output_dict = [{'raw_text':k, 'extracted':v} for k,v in zip([sentence], outputs)]
        else:
            output_dict = [{'raw_text':k, 'extracted':v} for k,v in zip(sentence, outputs)]

            
        if type(sentence) == str:
            return output_dict[0]
        
        return output_dict


if __name__ == '__main__':
    extractor = T5MovieExtractor('./checkpoints/generative_movie_title_ner')
    extractor('I love the scene in Titanic!', disable_tqdm=True)
    extractor(['I love the scene in Titanic!', 'Gone With the Wind\nAvengers End Game\The Zippers Saga'], disable_tqdm=True)
    extractor(
        ['I love the scene in Titanic!', 'Gone With the Wind\nAvengers End Game\The Zippers Saga', 'omg love this!'],
        batch_size=2
    )