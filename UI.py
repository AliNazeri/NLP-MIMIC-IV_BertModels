# -*- coding: UTF-8 -*-
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tracemalloc import start
import pandas as pd
import tkinter as tk
from keras.models import load_model
from transformers import TFBertModel
from transformers import AutoTokenizer
from nltk.corpus import stopwords
import contractions
from nltk.stem import WordNetLemmatizer
import regex as re

class app:
    def __init__(self, master):
        self.master = master
        self.master.geometry("700x550")
        self.master.configure(bg='#219ebc')
        self.master.title("TOP ICD 10")
        self.pfont_label = ('B Nazanin',12,'bold')
        self.efont_label = ('Arial',10,'bold')
        self.efont_label2 = ('Arial',8,'bold')
        self.efont_entry = ('Arial',12)
        self.pfont_button = ('B Nazanin',10,'bold')
        self.pfont_entry = ('B Nazanin',12)
        self.pfont_check = ('B Nazanin',10,'bold')
        self.model = ""
        self.models_loaded = False
        self.waitpage()
    
    def loading_models(self):
        self.bert_general_model = load_model('F:/project related/trained models/bert general/kaggle/working/BiomedVLP-CXR-BERT-general-16',custom_objects={"TFBertModel": TFBertModel})
        self.bert_abstract_model = load_model('F:/project related/trained models/abstract/kaggle/working/BiomedNLP-PubMedBERT-base-uncased-abstract-13',custom_objects={"TFBertModel": TFBertModel})
        self.bert_blue_model = load_model('F:/project related/trained models/bluebert/kaggle/working/bluebert-16',custom_objects={"TFBertModel": TFBertModel})

    def loading_tokenizers(self):
        self.bert_general_tokenizer = AutoTokenizer.from_pretrained("F:/project related/trained models/bert general", use_fast=False)
        self.bert_abstract_tokenizer = AutoTokenizer.from_pretrained("F:/project related/trained models/abstract", use_fast=False)
        self.bert_blue_tokenizer = AutoTokenizer.from_pretrained("F:/project related/trained models/bluebert", use_fast=False)

    def waitpage(self):
        def import_model():
            print("Models are loading please wait:")
            self.loading_models()
            print("Models loaded")
            self.loading_tokenizers()
            print("Tokenizers loaded")
            self.models_loaded = True
        
        def set_model(name):
            self.model = name
            self.mainpage()

        for i in self.master.winfo_children():
            i.destroy()
        self.frame1 = Frame(self.master,bg = '#219ebc')
        self.frame1.pack()
        
        self.b1 = Button(self.frame1, command=lambda: set_model('general'), text = "BERT-general", font=self.pfont_button, width=20, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 0, column = 0, sticky = W, pady = (5,5),padx=(5,5),columnspan=2)
        self.b2 = Button(self.frame1, command=lambda: set_model('abstract'), text = "Abstract-PubMedBERT", font=self.pfont_button, width=20, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 2, column = 0, sticky = W, pady = (5,5),padx=(5,5),columnspan=2)
        self.b2 = Button(self.frame1, command=lambda: set_model('blue'), text = "BlueBERT", font=self.pfont_button, width=20, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 4, column = 0, sticky = W, pady = (5,5),padx=(5,5),columnspan=2)

        if self.models_loaded == False:
            import_model()
    
    def mainpage(self):
        def openfile(text_place,pb):
            #inputs = self.e1.get("1.0",'end-1c')
            tf = filedialog.askopenfilename(
            initialdir="C:/Users/MainFrame/Desktop/", 
            title="Open Text file", 
            filetypes=(("Text Files", "*.txt"),)
            )
            
            if len(tf) > 0:
                file = open(tf,'r')
                opened = file.read()
                self.put_intobox(text_place,opened)
                do_the_math(opened,pb)

        def do_the_math(txt,pb):
            preprocessed_txt = preprocessing(txt)
            res = predict(preprocessed_txt)
            print(res)
            set_labels(res,pb)

        def preprocessing(txt):
            txt = txt.lower()
            txt = re.sub(r'\n','', txt)
            txt = re.sub(r'name no admission date discharge date', '', txt)
            txt = re.sub(r'date of birth', '', txt)
            txt = re.sub(r'name unit no admission date discharge date', '', txt)

            txt = contractions.fix(txt)

            mystpwrd = stopwords.words('english')
            mystpwrd.remove('not')
            txt = " ".join([word for word in str(txt).split() if word not in mystpwrd])

            lemmatizer = WordNetLemmatizer()
            txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()])

            txt = re.sub(' +',' ',txt)

            return txt
        
        def predict(txt):
            labels = ['I10', 'E785', 'Z87891', 'K219', 'F329', 'I2510', 'F419', 'N179','Z794', 'E039']

            if self.model == "blue":
                x = self.bert_blue_tokenizer(
                        txt,
                        add_special_tokens=True,
                        max_length= 512,
                        padding = 'max_length',
                        return_token_type_ids= False,
                        return_attention_mask= True,
                        truncation= True,
                        return_tensors = 'np')

                vector = self.bert_blue_model.predict({'input_ids':x['input_ids'],'attention_mask':x['attention_mask']})
                # Label 1: 0.421903520822525
                # Label 2: 0.4115627706050873
                # Label 3: 0.33502262830734253
                # Label 4: 0.28258219361305237
                # Label 5: 0.22734390199184418
                # Label 6: 0.21220369637012482
                # Label 7: 0.19085118174552917
                # Label 8: 0.1919291913509369
                # Label 9: 0.1449456363916397
                # Label 10: 0.14179493486881256
                thresholds = [0.421903520822525,0.3615627706050873,0.27502262830734253,0.28258219361305237,0.22734390199184418
                            ,0.21220369637012482,0.15085118174552917,0.1919291913509369,0.1449456363916397,0.14179493486881256]

            elif self.model == "general":
                x = self.bert_general_tokenizer(
                        txt,
                        add_special_tokens=True,
                        max_length= 512,
                        padding = 'max_length',
                        return_token_type_ids= False,
                        return_attention_mask= True,
                        truncation= True,
                        return_tensors = 'np')

                vector = self.bert_general_model.predict({'input_ids':x['input_ids'],'attention_mask':x['attention_mask']})
                # Label 1: 0.4852258563041687
                # Label 2: 0.45131203532218933
                # Label 3: 0.3432055413722992
                # Label 4: 0.2905704975128174
                # Label 5: 0.2269510179758072
                # Label 6: 0.179512619972229
                # Label 7: 0.17128156125545502
                # Label 8: 0.18445748090744019
                # Label 9: 0.13755233585834503
                # Label 10: 0.1571851372718811
                thresholds = [0.4852258563041687,0.45131203532218933,0.3432055413722992,0.2905704975128174,0.2269510179758072,0.179512619972229
                            ,0.17128156125545502,0.18445748090744019,0.13755233585834503, 0.1571851372718811]

            elif self.model == "abstract":
                x = self.bert_abstract_tokenizer(
                        txt,
                        add_special_tokens=True,
                        max_length= 512,
                        padding = 'max_length',
                        return_token_type_ids= False,
                        return_attention_mask= True,
                        truncation= True,
                        return_tensors = 'np')

                vector = self.bert_abstract_model.predict({'input_ids':x['input_ids'],'attention_mask':x['attention_mask']})
                # Label 1: 0.4252188503742218
                # Label 2: 0.43179163336753845
                # Label 3: 0.36237621307373047
                # Label 4: 0.27204763889312744
                # Label 5: 0.2108398675918579
                # Label 6: 0.22177961468696594
                # Label 7: 0.1640295386314392
                # Label 8: 0.24510458111763
                # Label 9: 0.14194940030574799
                # Label 10: 0.1622004508972168
                thresholds = [0.4252188503742218, 0.43179163336753845,0.36237621307373047,0.27204763889312744,0.2108398675918579,
                            0.22177961468696594,0.1640295386314392,0.24510458111763,0.14194940030574799,0.1622004508972168]
            
            binary_predictions = (vector >= thresholds).astype(int)
            return binary_predictions
        
        def set_labels(vector,pb):
            for i in range(10):
                pb[i]['value']= (100 * vector[0][i])
        
        for i in self.master.winfo_children():
            i.destroy()

        print(self.model)
        self.frame2 = Frame(self.master,bg = '#219ebc')
        self.frame2.pack()
        inp_string = tk.StringVar()
        self.b1 = Button(self.frame2, command=lambda: openfile(self.e1,pb), text = "خواندن متن", font=self.pfont_button, width=10, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 0, column = 0, sticky = W, pady = (5,5),padx=(5,5),columnspan=1)
        self.b2 = Button(self.frame2, command=self.waitpage, text = "بازگشت", font=self.pfont_button, width=10, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 0, column = 4, sticky = E, pady = (5,5),padx=(5,5),columnspan=1)
        self.le = Label(self.frame2, text = "محل وارد کردن متن ترخیص",bg = "#219ebc",fg='#03045e',font=self.pfont_label).grid(row = 1, column = 4, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.e1 = Text(self.frame2, width = 25,height=20,font=self.efont_entry, fg='#03045e',wrap=WORD)
        self.e1.grid(row = 2, column = 3, sticky = W, pady = (5,5),padx=20,rowspan=8,columnspan=3)
        self.b3 = Button(self.frame2, command=lambda: do_the_math(self.e1.get("1.0", "end-1c"),pb), text = "تایید", font=self.pfont_button, width=10, bg='#1d3557', fg='#a8dadc', activebackground='#a8dadc', activeforeground='#03045e').grid(row = 10, column = 4, sticky = E, pady = (5,5),padx=(5,5),columnspan=1)

        self.l1 = Label(self.frame2, text = "I10",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 2, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l2 = Label(self.frame2, text = "E785",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 3, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l3 = Label(self.frame2, text = "Z87891",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 4, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l4 = Label(self.frame2, text = "K219",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 5, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l5 = Label(self.frame2, text = "F329",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 6, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l6 = Label(self.frame2, text = "I2510",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 7, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l7 = Label(self.frame2, text = "F419",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 8, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l8 = Label(self.frame2, text = "N179",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 9, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l9 = Label(self.frame2, text = "Z794",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 10, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
        self.l10 = Label(self.frame2, text = "E039",bg = "#219ebc",fg='#03045e',font=self.efont_label).grid(row = 11, column = 0, sticky = W, pady = (5,5),padx=10,columnspan=1)
    
        self.l11 = Label(self.frame2, text = "Essential (primary) hypertension",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 2, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l22 = Label(self.frame2, text = "Hyperlipidemia, unspecified",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 3, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l33 = Label(self.frame2, text = "Personal history of\nnicotine dependence",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 4, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l44 = Label(self.frame2, text = "Gastro-esophageal reflux disease",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 5, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l55 = Label(self.frame2, text = "Major depressive disorder",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 6, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l66 = Label(self.frame2, text = "Atherosclerotic heart disease of\nnative coronary artery",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 7, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l77 = Label(self.frame2, text = "Anxiety disorder, unspecified",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 8, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l88 = Label(self.frame2, text = "Acute kidney failure, unspecified",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 9, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l99 = Label(self.frame2, text = "Long term (current) use of insulin",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 10, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)
        self.l100 = Label(self.frame2, text = "Hypothyroidism, unspecified",bg = "#219ebc",fg='#03045e',font=self.efont_label2).grid(row = 11, column = 2, sticky = W, pady = (5,5),padx=10,columnspan=1,rowspan=1)

        self.pb1 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb1.grid(row = 2, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb2 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb2.grid(row = 3, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb3 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb3.grid(row = 4, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb4 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb4.grid(row = 5, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb5 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb5 .grid(row = 6, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb6 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb6.grid(row = 7, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb7 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb7.grid(row = 8, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb8 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb8.grid(row = 9, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb9 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb9.grid(row = 10, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)
        self.pb10 = ttk.Progressbar(self.frame2, length=100,mode='determinate')
        self.pb10.grid(row = 11, column = 1, sticky = W, pady = (5,5),padx=2,columnspan=1)

        pb = [self.pb1,self.pb2,self.pb3,self.pb4,self.pb5,self.pb6,self.pb7,self.pb8,self.pb9,self.pb10]

    def put_intobox(self,lb,arr):
        if len(lb.get("1.0",END)) > 1:
            lb.delete("1.0", tk.END)
        lb.insert(END,arr)

root = Tk()
app(root)
root.mainloop()
