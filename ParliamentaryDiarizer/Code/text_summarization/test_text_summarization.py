
from text_summarizers import HuggingFaceTextSummarizer, NewHuggingFaceTextSummarizer, Es2EnTranslator, En2EsTranslator


# with open("/home/jorge/TFG/ParliamentaryDiarizer/Code/text_summarization/text_to_summarize.txt", "r") as file:
#     text = file.read()


# text_summarizer = HuggingFaceTextSummarizer()
# print("")
# print(text_summarizer.summarize_text(text))

# # print("")
# # text_summarizer = HuggingFaceTextSummarizer(model="mrm8488/bert2bert_shared-spanish-finetuned-summarization")
# # print("")
# # print(text_summarizer.summarize_text(text))

# print("")
# text_summarizer = NewHuggingFaceTextSummarizer()
# print("")
# print(text_summarizer.summarize_text(text))


print("")
print(Es2EnTranslator().translate("Me llamo Jorge, tengo 21 años y me gustan los perros."))
print(En2EsTranslator().translate("My name is Jorge, I'm 21 years old and I like dogs."))