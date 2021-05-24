from summarizer import Summarizer

DATA_DIR = './data/plaintext/Chinh tri'
SUMMARY_DIR = './data/bert_summary/Chinh tri'

for i in range(1, 32):
    if i < 10:
        filename = DATA_DIR + "/" + "CT0" + str(i) + ".txt"
    else:
        filename = DATA_DIR + "/" + "CT" + str(i) + ".txt"
    
    title = ""
    paragraph = ""
    
    with open(filename, encoding='utf-8') as f:
        title = f.readline()
        for line in f:
            paragraph += line.strip()

    model = Summarizer()
    result = ''.join(model(body=paragraph, ratio=0.4, min_length=30))

    if i < 10:
        filename_output = SUMMARY_DIR + "/" + "CT0" + str(i) + ".txt"
    else:
        filename_output = SUMMARY_DIR + "/" + "CT" + str(i) + ".txt"


    with open(filename_output, "w", encoding='utf-8') as f:
        f.write(result)
    
    print(i)
