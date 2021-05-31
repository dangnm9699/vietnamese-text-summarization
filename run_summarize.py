from summarizer import Summarizer

DATA_DIR = './data/plaintext/Xa Hoi'
SUMMARY_DIR = './data/bert_summary/Xa Hoi'

model = Summarizer()

for i in range(1, 36):
    if i < 10:
        filename = DATA_DIR + "/" + "XH0" + str(i) + ".txt"
    else:
        filename = DATA_DIR + "/" + "XH" + str(i) + ".txt"
    print(filename)

    title = ""
    paragraph = ""

    with open(filename, encoding='utf-8') as f:
        title = f.readline()
        for line in f:
            paragraph += line.strip()

    result = ''.join(model(body=paragraph, ratio=0.4, min_length=30))
    result = result.replace('_', ' ')

    if i < 10:
        filename_output = SUMMARY_DIR + "/" + "XH0" + str(i) + ".txt"
    else:
        filename_output = SUMMARY_DIR + "/" + "XH" + str(i) + ".txt"

    with open(filename_output, "w", encoding='utf-8') as f:
        f.write(result)

    print(i)
