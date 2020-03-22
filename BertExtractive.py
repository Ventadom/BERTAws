from summarizer import Summarizer
import os
from nltk import sent_tokenize, word_tokenize
import re 
import numpy as np
import rouge
import time

def textWordCount(Text):
    number_of_words = word_tokenize(Text)
    count=(len(number_of_words))
    return count

def textSentenceCount(Text):
    number_of_sentences = sent_tokenize(Text)
    count=(len(number_of_sentences))
    return count

def findRatio(count,ratio):
    if(count*ratio<950):
        Ratio=np.round(ratio,3)
        return Ratio
    else:
        return findRatio(count,ratio-0.001)
    
def findRatioLong(count,ratio,limit):
    if(count*ratio<limit):
        Ratio=np.round(ratio,3)
        return Ratio
    else:
        return findRatioLong(count,ratio-0.001,limit)
    
def shortSentenceClean(Corpus,Limit):
    sentences = sent_tokenize(Corpus)
    for sentence in sentences:
        if (textWordCount(sentence)<Limit):
            sentences.remove(sentence)
#            print("DELETED SHORT SENTENCE, SIZE:",textWordCount(sentence))
#            print(sentence)
            
    newCorpus = ' '.join(sentences)
    return newCorpus

def longSentenceClean(Corpus,Limit):
    sentences = sent_tokenize(Corpus)
    for sentence in sentences:
        if (textWordCount(sentence)>Limit):
            sentences.remove(sentence)
#            print("****DELETED LONG SENTENCE, SIZE:",textWordCount(sentence))
#            print(sentence)
    
    newCorpus = ' '.join(sentences)
    return newCorpus

def cleanTitle(Corpus):
    Corpus = Corpus.replace("\n"," ")
    sentences = sent_tokenize(Corpus)
    for sentence in sentences:
        count=0
        for i in sentence:
            if(i.isupper()):
                count=count+1
        if(count*3>len(sentence)):
            sentences.remove(sentence)
            #print(sentence)
            #print("**** DELETED TITLE, SIZE:",textWordCount(sentence))
            #print(sentence)
    
    newCorpus = ' '.join(sentences)
    return newCorpus

def cleanManyCharacterandNumber(Corpus,rate):
    Corpus = Corpus.replace("\n"," ")
    sentences = sent_tokenize(Corpus)
    for sentence in sentences:        
        digit=letter=other=0
        for c in sentence:
            if c.isdigit():
                digit=digit+1
            elif c.isalpha():
                letter=letter+1
            else:
                other=other+1
        
        if(len(sentence)<(digit*rate)):
            sentences.remove(sentence)
#            print("****DELETED NUMBER, SIZE:",textWordCount(sentence))
#            print(sentence)
            continue
        
        if(len(sentence)<(other-sentence.count(' '))*10):
            sentences.remove(sentence)
#            print("****DELETED CHARACTER, SIZE:",textWordCount(sentence))
#            print(sentence)
        
    newCorpus = ' '.join(sentences)
    return newCorpus

def cleanParenthesis(Corpus):
    Corpus = Corpus.replace("\n"," ")
    cleaned=[]
    sentences = sent_tokenize(Corpus)
    for sentence in sentences:
        try:
            temp =re.sub("[\(\[].*?[\)\]]", "", sentence) #remove () and []
            cleaned.append(temp)
            #print(sentence)
            #print("---------")
            #print(temp)
            #print("---------------------------------------------")
            continue
        except:
            pass
    
    newCorpus = ' '.join(cleaned)
    #print("---------------------------------------------")
    return newCorpus

def preProcessingText(Corpus):
    
    cleanedText = Corpus.replace("\n"," ")    
    cleanedText=cleanParenthesis(cleanedText)
    cleanedText=cleanManyCharacterandNumber(cleanedText,10)    
    cleanedText=shortSentenceClean(cleanedText,7)
    cleanedText=longSentenceClean(cleanedText,80)
    
    return cleanedText
    
def preProcessingTextLong(Corpus):
    
    cleanedText = Corpus.replace("\n"," ")    
    cleanedText=cleanManyCharacterandNumber(cleanedText,12)    
    cleanedText=shortSentenceClean(cleanedText,6)
    cleanedText=longSentenceClean(cleanedText,40)
    cleanedText=cleanTitle(cleanedText)
    
    return cleanedText

startTimeforOverall = time.time()
inputs = os.listdir('annual_reports/')
cleanedReports=[]
DocumentSentenceCounts=[]
DocumentWordCounts=[]
AfterPreprocessingDocumentSentenceCounts=[]
AfterPreprocessingDocumentWordCounts=[]
SummarySentenceCounts=[]
SummaryWordCounts=[]

for x in range(len(inputs)):
    startTimeforDocument = time.time()
    #read files
    print('{0}. Document'.format(len(cleanedReports)+1))
    print('File Name:{0}'.format(inputs[x]))
    sourceFilePath='annual_reports/' + inputs[x]
    file = open(sourceFilePath, encoding="utf8")
    temp=file.read()
    file.close()
    
    OrginalTextSentenceCount=textSentenceCount(temp)
    OrginalTextWordCount=textWordCount(temp)
    print('Orginal Text Sentence Count:{0}, Orginal Text Word Count:{1}'.format(OrginalTextSentenceCount,OrginalTextWordCount))
    DocumentSentenceCounts.append(OrginalTextSentenceCount)
    DocumentWordCounts.append(OrginalTextWordCount)
    cleanedReport=preProcessingText(temp)
    
    
    PreprocessedTextLenght=len(cleanedReport)
    print('Text lenght before BERT:{0}'.format(PreprocessedTextLenght))
    
    if(PreprocessedTextLenght>1000000):
        cleanedReport=preProcessingTextLong(cleanedReport)
        PreprocessedTextLenght=len(cleanedReport)
        print('Second Text lenght before BERT:{0}'.format(PreprocessedTextLenght))
    
    cleanedReports.append(cleanedReport)
    PreprocessedTextSentenceCount=textSentenceCount(cleanedReport)
    PreprocessedTextWordCount=textWordCount(cleanedReport)
    #print(cleanedReport)
    print('Preprocessed Text Sentence Count:{0}, Preprocessed Text Word Count:{1}'.format(PreprocessedTextSentenceCount,PreprocessedTextWordCount))
    AfterPreprocessingDocumentSentenceCounts.append(PreprocessedTextSentenceCount)
    AfterPreprocessingDocumentWordCounts.append(PreprocessedTextWordCount)
    
    #BERT

    Ratio=findRatio(PreprocessedTextWordCount,1)
    print('Ratio:',Ratio)
    if (Ratio<=0.0):
        Ratio=0.001
        print('Second Ratio:',Ratio)
    model = Summarizer()
    result = model(cleanedReport,ratio=Ratio)
    summary = ''.join(result)
    #write
    destinationFilePath='summaries/' + inputs[x]
    file2 = open(destinationFilePath,"w",encoding='utf-8') 
    file2.write(summary)
    file2.close()
    SummaryTextSentenceCount=textSentenceCount(summary)
    SummaryTextWordCount=textWordCount(summary)
    print('Summary Sentence Count:{0}, Summary Word Count:{1}'.format(SummaryTextSentenceCount,SummaryTextWordCount))
    SummarySentenceCounts.append(SummaryTextSentenceCount)
    SummaryWordCounts.append(SummaryTextWordCount)
    
    Limit=930
    while True:        
        if(SummaryTextWordCount>999):
            if(SummaryTextWordCount>900 and SummaryTextWordCount<1100):
                Ratio=Ratio-0.001
            else:
                Ratio=Ratio-0.003
            Ratio=findRatioLong(PreprocessedTextWordCount,Ratio,Limit)
            print('Summary Word Limit:',Limit)
            print('New Ratio:',Ratio)
            if (Ratio<=0.0):
                Ratio=0.001
                print('New Ratio:',Ratio)
            model = Summarizer()
            result = model(cleanedReport,ratio=Ratio)
            summary = ''.join(result)
            #write
            destinationFilePath='summaries/' + inputs[x]
            file2 = open(destinationFilePath,"w",encoding='utf-8') 
            file2.write(summary)
            file2.close()
            SummaryTextSentenceCount=textSentenceCount(summary)
            SummaryTextWordCount=textWordCount(summary)
            print('New Summary Sentence Count:{0}, Summary Word Count:{1}'.format(SummaryTextSentenceCount,SummaryTextWordCount))
            SummarySentenceCounts.pop()
            SummaryWordCounts.pop()
            SummarySentenceCounts.append(SummaryTextSentenceCount)
            SummaryWordCounts.append(SummaryTextWordCount)
            Limit=Limit-20
        else:
            break
    
    elapsedTimeforDocument = time.time() - startTimeforDocument
    elapsedTimeforAll = time.time() - startTimeforOverall
    print('Document processing time: '+time.strftime("%M:%S", time.gmtime(elapsedTimeforDocument)))
    print('Total processing time: '+time.strftime("%d:%H:%M:%S", time.gmtime(elapsedTimeforAll)))
    print("######################################################################################")

def Average(lst): 
    return sum(lst) / len(lst) 

print("#################################################################")
averageDocumentSentenceCounts = Average(DocumentSentenceCounts)
print("averageDocumentSentenceCounts:",averageDocumentSentenceCounts)

averageDocumentWordCounts = Average(DocumentWordCounts)
print("averageDocumentWordCounts:",averageDocumentWordCounts)

print("#################################################################")
averageAfterPreprocessingDocumentSentenceCounts = Average(AfterPreprocessingDocumentSentenceCounts)
print("averageAfterPreprocessingDocumentSentenceCounts:",averageAfterPreprocessingDocumentSentenceCounts)

averageAfterPreprocessingDocumentWordCounts = Average(AfterPreprocessingDocumentWordCounts)
print("averageAfterPreprocessingDocumentWordCounts:",averageAfterPreprocessingDocumentWordCounts)

print("#################################################################")
averageSummarySentenceCounts = Average(SummarySentenceCounts)
print("averageSummarySentenceCounts:",averageSummarySentenceCounts)

averageSummaryWordCounts = Average(SummaryWordCounts)
print("averageSummaryWordCounts:",averageSummaryWordCounts)

print("####################################################################################")

import os
entries1 = os.listdir('annual_reports')
entries2 = os.listdir('summaries')

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


for aggregator in ['Avg']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    references=[]

    for x in range(len(entries2)):
        tempfilepath='summaries\\' + entries2[x]
        file = open(tempfilepath, encoding="utf8")
        references.append(file.read())
        
    
    hypothesis=[]

    for x in range(len(entries2)):
        tempfilepath='annual_reports\\' + entries2[x]
        file = open(tempfilepath, encoding="utf8")
        hypothesis.append(file.read())
    
    scores = evaluator.get_scores(hypothesis, references)


    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()
    
print("#################################################################")
      
goldreferences=[]

for x in range(len(entries2)):
    try:
        tempfilepathR='gold_summaries\\' + entries2[x].split(".")[0]+'_1.txt'
        fileR = open(tempfilepathR, encoding="utf8")
        goldreferences.append(fileR.read())


    except:
        print("Missing : ", tempfilepathR)


for aggregator in ['Avg']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        
    scores = evaluator.get_scores(hypothesis, goldreferences)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()
    
tempListSentences=[]
tempListWords=[]

for i in range(len(goldreferences)):
    tempListWords.append(textWordCount(goldreferences[i]))
    tempListSentences.append(textSentenceCount(goldreferences[i]))
    

averagetempListSentences = Average(tempListSentences)
print("average gold Summaries_1 Sentence Count:",averagetempListSentences)

averagetempListWords = Average(tempListWords)
print("average gold Summaries_1 Words Count:",averagetempListWords)
print("####################################################################################")
      
goldreferences=[]

for x in range(len(entries2)):
    try:
        tempfilepathR='gold_summaries\\' + entries2[x].split(".")[0]+'_2.txt'
        fileR = open(tempfilepathR, encoding="utf8")
        goldreferences.append(fileR.read())


    except:
        print("Missing : ", tempfilepathR)


for aggregator in ['Avg']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        
    scores = evaluator.get_scores(hypothesis, goldreferences)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()

tempListSentences=[]
tempListWords=[]

for i in range(len(goldreferences)):
    tempListWords.append(textWordCount(goldreferences[i]))
    tempListSentences.append(textSentenceCount(goldreferences[i]))
    
averagetempListSentences = Average(tempListSentences)
print("average gold Summaries_2 Sentence Count:",averagetempListSentences)
averagetempListWords = Average(tempListWords)
print("average gold Summaries_2 Words Count:",averagetempListWords)
print("####################################################################################")

goldreferences=[]

for x in range(len(entries2)):
    try:
        tempfilepathR='gold_summaries\\' + entries2[x].split(".")[0]+'_3.txt'
        fileR = open(tempfilepathR, encoding="utf8")
        goldreferences.append(fileR.read())


    except:
        print("Missing : ", tempfilepathR)

for aggregator in ['Avg']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        
    scores = evaluator.get_scores(hypothesis, goldreferences)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()

tempListSentences=[]
tempListWords=[]

for i in range(len(goldreferences)):
    tempListWords.append(textWordCount(goldreferences[i]))
    tempListSentences.append(textSentenceCount(goldreferences[i]))
    

averagetempListSentences = Average(tempListSentences)
print("average gold Summaries_3 Sentence Count:",averagetempListSentences)

averagetempListWords = Average(tempListWords)
print("average gold Summaries_3 Words Count:",averagetempListWords)
print("####################################################################################")

import os
entries2 = os.listdir('summaries')
entries3 = os.listdir('gold_summaries')
hypothesisAll=[]
goldReferencesAll=[]
onlyName=[]

for item in range(len(entries2)):
    temp=entries2[item].split(".")[0]
    onlyName.append(temp)

count=0
for item in range(len(entries3)):
    if(entries3[item].split("_")[0] in onlyName):
        tempfilepathR='gold_summaries\\' + entries3[item]
        fileR = open(tempfilepathR, encoding="utf8")
        goldReferencesAll.append(fileR.read())
        fileR.close()

        tempfilepathH='annual_reports\\' + entries3[item].split("_")[0]+".txt"
        fileH = open(tempfilepathH, encoding="utf8")
        hypothesisAll.append(fileH.read())
        
for aggregator in ['Avg']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        
    scores = evaluator.get_scores(hypothesisAll, goldReferencesAll)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()

tempListSentences=[]
tempListWords=[]

for i in range(len(goldReferencesAll)):
    tempListWords.append(textWordCount(goldReferencesAll[i]))
    tempListSentences.append(textSentenceCount(goldReferencesAll[i]))
    

averagetempListSentences = Average(tempListSentences)
print("average gold Summaries All Sentence Count:",averagetempListSentences)

averagetempListWords = Average(tempListWords)
print("average gold Summaries All Words Count:",averagetempListWords)
print("####################################################################################")
    
     