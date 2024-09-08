from django.shortcuts import render
from transformers import pipeline, PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')

def chunk_text(text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk,skip_special_tokens=True) for chunk in chunks]

def summarize_text(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if not text:
            return render(request, 'app/index.html', {'error': 'No text provided'})
        
        try:
            chunks = chunk_text(text, max_length=1024)  
            
            summarizer = pipeline('summarization', model='google/pegasus-large')
            summaries = [summarizer(chunk, max_length=150, min_length=1, num_beams=4, length_penalty=1.0) for chunk in chunks]
            
            combined_summary = ' '.join([summary[0]['summary_text'] for summary in summaries])
            return render(request, 'app/index.html', {'summary': combined_summary, 'text': text})
        except Exception as e:
            print(f"Error: {e}")
            return render(request, 'app/index.html', {'error': 'An error occurred during summarization'})
    return render(request, 'app/index.html')
