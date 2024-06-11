
import sys
import os
from abc import ABC, abstractmethod

import openai
import torch
from transformers import pipeline, BertTokenizerFast, EncoderDecoderModel, PegasusTokenizer, PegasusForConditionalGeneration
from openai import OpenAI

sys.path.append(os.getcwd() + "/Code")
import Code.config as config


class TextSummarizer(ABC):

    @abstractmethod
    def summarize_text(self, text):
        pass

    def _set_min_length(self, text):
        self._min_length = len(text.split()) // 4

    def _set_max_length(self, text):
        self._max_length = len(text.split()) // 2


class GPTTextSummarizer(TextSummarizer):

    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def summarize_text(self, text):

        try:
            response = self._client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Necesito que me generes un resumen de una intervención en un parlamento. Dicha intervención es la siguiente: " + text}]
                        )
            return response.choices[0].message.content

        except openai.APIError:
            return "No se pudo devolver el resumen por error en la API. Compruebe la conexión a red y reinténtelo."

        except openai.Timeout:
            return "No se pudo devolver el resumen por exceso de tiempo."



class HuggingFaceTextSummarizer(TextSummarizer):

    def __init__(self, model=None):
        self._max_length = 130
        self._min_length = 30

        if model != None:
            self._model = pipeline("summarization", 
                               min_length=self._min_length,
                               max_length=self._max_length, 
                               do_sample=False,
                               device = "cuda" if torch.cuda.is_available() else "cpu",
                               model=model)
        else:
            self._model = pipeline("summarization", 
                               min_length=self._min_length,
                               max_length=self._max_length, 
                               do_sample=False,
                               device = "cuda" if torch.cuda.is_available() else "cpu")
        
    def summarize_text(self, text):
        summ_dict = self._model(text)
        return summ_dict[0]["summary_text"]
    


class MRMSpanishFineTunedTextSummarizer(TextSummarizer):

    def __init__(self, model=None):
        self._max_length = None
        self._tokenizer = BertTokenizerFast.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-summarization")
        self._model = EncoderDecoderModel.from_pretrained(
            "mrm8488/bert2bert_shared-spanish-finetuned-summarization").to("cuda" if torch.cuda.is_available() else "cpu")

    def summarize_text(self, text):
        self._set_max_length(text)
        inputs = self._tokenizer([text], padding="max_length", truncation=True, max_length=self._max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = inputs.attention_mask.to("cuda" if torch.cuda.is_available() else "cpu")
        output = self._model.generate(input_ids, attention_mask=attention_mask)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


#######################################################################


class Lang2LangTranslator(ABC):

    @abstractmethod
    def translate(self, text):
        pass


class Es2EnTranslator(Lang2LangTranslator):

    def __init__(self):
        self._pipeline = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en", device="cuda")

    def translate(self, text):
        return self._pipeline(text)[0]["translation_text"]


class En2EsTranslator(Lang2LangTranslator):

    def __init__(self):
        self._pipeline = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device="cuda")

    def translate(self, text):
        return self._pipeline(text)[0]["translation_text"]
    


class TranslatingHuggingFaceTextSummarizer(TextSummarizer):

    def __init__(self, model=None):
        self._max_length = 130
        self._min_length = 30

        if model != None:
            self._model = pipeline("summarization", 
                               min_length=self._min_length,
                               max_length=self._max_length, 
                               do_sample=False,
                               device = "cuda" if torch.cuda.is_available() else "cpu",
                               model=model)
        else:
            self._model = pipeline("summarization", 
                               min_length=self._min_length,
                               max_length=self._max_length, 
                               do_sample=False,
                               device = "cuda" if torch.cuda.is_available() else "cpu")
            
        self._es_2_en_translator = Es2EnTranslator()
        self._en_2_es_translator = En2EsTranslator()
        self._chunk_size = 512
        

    def _es_2_en_translation(self, text): # Español a ingles
        tokens = self._tokenizer.tokenize(text)
        amount_chunks = len(tokens) // self._chunk_size
        chunks = [tokens[self._chunk_size*i:self._chunk_size*(i+1)] for i in range(amount_chunks)]
        chunks.append(tokens[self._chunk_size*amount_chunks:]) # El chunk restante
        string_chunks = [self._tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        return [self._es_2_en_translator.translate(string_chunk) for string_chunk in string_chunks]
    
    def _en_2_es_translation(self, text): # Inglés a español
        tokens = self._tokenizer.tokenize(text)
        amount_chunks = len(tokens) // self._chunk_size
        chunks = [tokens[self._chunk_size*i:self._chunk_size*(i+1)] for i in range(amount_chunks)]
        chunks.append(tokens[self._chunk_size*amount_chunks:]) # El chunk restante
        string_chunks = [self._tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        return [self._en_2_es_translator.translate(string_chunk) for string_chunk in string_chunks]


    def summarize_text(self, text):
        summ_dict = self._model(text)
        return summ_dict[0]["summary_text"]
    



class PegasusTextSummarizer(TextSummarizer):

    def __init__(self):
        self._model_name = "google/pegasus-xsum"
        self._tokenizer = PegasusTokenizer.from_pretrained(self._model_name)
        self._model = PegasusForConditionalGeneration.from_pretrained(self._model_name).to("cuda")
        self._length_penalty = 0.4 # Valores más pequeños para secuencias más cortas, y más grandes para más largas
        self._num_beans = 5
        self._es_2_en_translator = Es2EnTranslator()
        self._en_2_es_translator = En2EsTranslator()
        self._chunk_size = 512
        self._min_length = None
        self._max_length = None


    def _es_2_en_translation(self, text): # Español a ingles
        tokens = self._tokenizer.tokenize(text)
        amount_chunks = len(tokens) // self._chunk_size
        chunks = [tokens[self._chunk_size*i:self._chunk_size*(i+1)] for i in range(amount_chunks)]
        chunks.append(tokens[self._chunk_size*amount_chunks:]) # El chunk restante
        string_chunks = [self._tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        return [self._es_2_en_translator.translate(string_chunk) for string_chunk in string_chunks]
    

    def _en_2_es_translation(self, text): # Inglés a español
        tokens = self._tokenizer.tokenize(text)
        amount_chunks = len(tokens) // self._chunk_size
        chunks = [tokens[self._chunk_size*i:self._chunk_size*(i+1)] for i in range(amount_chunks)]
        chunks.append(tokens[self._chunk_size*amount_chunks:]) # El chunk restante
        string_chunks = [self._tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        return [self._en_2_es_translator.translate(string_chunk) for string_chunk in string_chunks]
    

    def summarize_text(self, text):

        # CON TRADUCCIÓN
        english_translated_intervention = self._es_2_en_translation(text)[0]

        # Establecemos tamaño mínimo y máximo del resumen en función del tamaño del texto original
        self._set_min_length(english_translated_intervention)
        self._set_max_length(english_translated_intervention)

        inputs = self._tokenizer.encode(english_translated_intervention, 
                                        return_tensors="pt", padding="longest", truncation=True).to("cuda")
        summary_ids = self._model.generate(inputs, 
                                           max_length=self._max_length, 
                                           min_length=self._min_length, 
                                           length_penalty=self._length_penalty, 
                                           num_beams=self._num_beans, 
                                           early_stopping=True)
        english_summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        spanish_summary = self._en_2_es_translation(english_summary)[0]

        # SIN TRADUCCIÓN
        # inputs = self._tokenizer.encode(text, 
        #                                 return_tensors="pt", padding="longest", truncation=True).to("cuda")
        # summary_ids = self._model.generate(inputs, 
        #                                    max_length=self._max_length, 
        #                                    min_length=self._min_length, 
        #                                    length_penalty=self._length_penalty, 
        #                                    num_beams=self._num_beans, 
        #                                    early_stopping=True)
        # spanish_summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return spanish_summary