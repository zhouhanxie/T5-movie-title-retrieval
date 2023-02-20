import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    """https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class T5MovieExtractor:
    
    def __init__(
        self, 
        model_name_or_path,
        device='cuda:0', 
        max_length=148, 
        prompt = 'Perform named entity recognition to extract movie titles. '\
        'List all movie titles that appears in text above. '\
        'Separate movie names using comma, and do not include years, genre, director, award '\
        'and other movie-related word that are not titles.'
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)
        self.prompt = prompt
        
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

            output = self.model.generate(**encoded_src)
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        output_dict = [{'raw_text':k, 'extracted':v} for k,v in dict(zip(raw_inp, output)).items()]
        
        return output_dict
    
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
            
        batches = list(chunks(raw_inp, batch_size))
        outputs = []
        
        for batch in tqdm(batches, disable=disable_tqdm):
            try:
                outputs += self._extract(batch)
            except KeyBoardInterrupt:
                raise
            except:
                if on_error == 'nothing':
                    pass
                elif on_error == 'echo':
                    print('---')
                    print('T5MovieExtractor: oops broken batch, sorry...')
                    print(batch)
                    print('---')
                raise ValueError('the batch above broke the pipeline')
            
        if type(sentence) == str:
            return outputs[0]
        return outputs


if __name__ == '__main__':
    extractor = T5MovieExtractor('./checkpoints/generative_movie_title_ner')
    extractor('I love the scene in Titanic!', disable_tqdm=True)
    extractor(['I love the scene in Titanic!', 'Gone With the Wind\nAvengers End Game\The Zippers Saga'], disable_tqdm=True)
    extractor(
        ['I love the scene in Titanic!', 'Gone With the Wind\nAvengers End Game\The Zippers Saga', 'omg love this!'],
        batch_size=2
    )