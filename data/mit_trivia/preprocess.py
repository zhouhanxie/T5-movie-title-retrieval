import pandas as pd

def list_split_by(list1, split_by=None):
    """
    https://stackoverflow.com/questions/71395055/python-split-a-list-into-several-lists-for-each-new-line-character
    """
    list2 = []
    tmp = []
    for item in list1:
        if item != split_by:
            tmp.append(item)
        else:
            #Note we aren't actually processing this item of the input list, as '\n' by itself is unwanted
            list2.append(tmp)
            tmp = []
    return list2

def main():
    with open('engtest.bio', 'r') as ifp:
        lines = ifp.readlines()
    datapoints = list_split_by(lines, '\n')
    text_snippets = []
    extracted_titles = []
    for datum in datapoints:
        labels = [i.strip().split('\t')[0] for i in datum]
        words = [i.strip().split('\t')[1] for i in datum]
        
        state = 'Nothing'
        titles = []
        tmp = []
        for i in range(len(words)):
            if labels[i] == 'B-TITLE':
                tmp.append(words[i])
                state = 'Parsing'
            elif labels[i] == 'I-TITLE':
                tmp.append(words[i])
            else:
                if state == 'Parsing':
                    titles.append(' '.join(tmp))
                    state = 'Nothing'
        text = ' '.join(words)
        text_snippets.append(text)
        extracted_titles.append(titles)

    extracted_titles = [','.join(i) for i in extracted_titles]
    extracted_titles = [',' if len(i) == 0 else i for i in extracted_titles ]


    df = pd.DataFrame()
    df['Input.selftext'] = text_snippets
    df['Answer.crowd_input'] = extracted_titles
    df.to_csv('mit_movie_testset.csv')
    print('done!')

if __name__ == '__main__':
    main()
