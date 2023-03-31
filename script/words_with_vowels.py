import numpy as np
import pandas as pd

words_freq = pd.read_csv('D:\Data\word_frequency_2006\COCA_60000_depuped.csv')
vowels = list('aeiou')
double_vowels = []
for v1 in vowels:
    for v2 in vowels:
        double_vowels.append(v1 + v2)

words = words_freq.dropna()
words_with_ao = words[words.word.str.contains('ao', na=False)]

"""
words with 'ao':

  freq_index             word
        2641    extraordinary
        3755            chaos
        7434  extraordinarily
        7533          chaotic
       17417          karaoke
       21964          pharaoh
       25120           maoist
       25686            cacao
       27815            aorta
       29056   intraoperative
       29155          laotian
       29629           aortic
       30279             ciao
       35484           kaolin
       36190              tao
       36266        pharaonic
       38575           taoist
       39942             gaol
       40778      chaotically
       43096           baobab
       44494           maoism
       45090            haole
       46240         post-mao
       47997      intraocular
       54234        intraoral
       56015    ultraorthodox
       57181              hao
       60021           gaoler
"""

dv = pd.DataFrame({
    'double_vowels': double_vowels,
    'words_count': np.zeros(len(double_vowels), dtype=int)})
for v in double_vowels:
    words_with_v = words[words.word.str.contains(v, na=False)]
    dv.loc[dv['double_vowels'] == v, 'words_count'] = len(words_with_v)
dv

"""
   double_vowels  words_count
0             aa           16
1             ae          121
2             ai         1203
3             ao           28
4             au          545
5             ea         2833
6             ee         1527
7             ei          480
8             eo          409
9             eu          282
10            ia         1817
11            ie         1230
12            ii           12
13            io         3479
14            iu          159
15            oa          546
16            oe          184
17            oi          539
18            oo         1478
19            ou         2649
20            ua          644
21            ue          599
22            ui          549
23            uo          120
24            uu            7
"""

most_rare = ['uu', 'ii', 'aa']
for v in most_rare:
    l = words[words.word.str.contains(v, na=False)].word.to_list()
    print('words with ' + v + ' (' + str(len(l)) + '):')
    print(l)
    print()

"""
words with uu (7):
['vacuum', 'continuum', 'muumuu', 'residuum', 'vacuuming', 'vacuum-packed', 'vacuum-sealed']

words with ii (12):
['skiing', 'hawaiian', 'shiite', 'shiitake', 'fasciitis', 'shiite-dominated', 'post-wwii', 'water-skiing', 'shiite-led', 'sunni-shiite', 'heli-skiing', 'freeskiing']

words with aa (16):
['bazaar', 'baathist', 'aah', 'maasai', 'afrikaans', 'canaanite', 'de-baathification', 'aaah', 'naan', 'non-gaap', 'salaam', 'aardvark', 'aaaah', 'laager', 'ujamaa', 'afrikaaner']
"""
