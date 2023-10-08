from flask import Flask, render_template, request
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
import torch


SummarEase = Flask(__name__, template_folder="templates")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


@SummarEase.route('/', methods=['GET','POST'])
def home():
    return render_template('index.html')

@SummarEase.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]
        tokenized_text = tokenizer(inputtext, max_length=1024, return_tensors="pt", truncation="True")
        summary_ids = model.generate(tokenized_text, num_beams=2, min_length=0, max_length=20)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # '''
    #         text = <start> i am depika <end>
    #         vocab = { i: 1, am : 2, yash: 3, start 4}

    #         token = [i, am , depika]
    #         encode = [1 2, 3, 4]

    #         summary_ids = [[4, 3,1, 5]]

    #         summary = depika i

        
    #     '''

    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__':
    SummarEase.run()
