import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np;
import os
import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split
import re
import argparse


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--no_gpu', dest='no_gpu', action='store_true')

parser.add_argument('--w2v', default='./datasets/text8.model',)
parser.add_argument('--w2v_self', dest='w2v_self', action='store_true')

parser.add_argument('--word_dim', default=20, type=int)

parser.add_argument('--use_random', dest='use_random', action='store_true')
parser.add_argument('--use_normal', dest='use_normal', action='store_true')

parser.add_argument('--save_every', default=100, type=int)
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
#parser.add_argument('--load', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
parser.add_argument('--name', default='text8')

args = parser.parse_args()


model_dir = parser.outf + parser.name + "/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if parser.no_gpu:
    device = torch.device("cpu")
else:
    #Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def words2line(words):
    line = ""
    for word in words:
        line = line + " " + word
    return line
    
#load word 2 vetc，加载词向量，可以事先预训练
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 150  # Dimensions for each word vector


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
def datahelper(dir):
    with open(dir, "r", encoding = 'utf-8') as f_csv:
        reviews = []
        labels = []
        ce = csv.reader(f_csv)
        rows = [row for row in ce]
        #reviews = ce[:,0]
        #lables = ce[:,1]
        texts = []
        rows = rows[1:]
        for line in rows:
        #    line = line.split(',')
        #    print(line)
            reviews.append(line[0])
            #print(line[0])
            #print(line[1])
            labels.append(float(line[1]))
    for line in reviews:
        #indexCounter = 0
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
#        print(split)
        texts.append(split)
    return texts, labels


#ids = np.zeros((num_files, max_seq_num), dtype='int32')
#file_count = 0



def getw2v(model_dir, sentences=None):
    #model_file_name = 'new_model_big.txt'
    # 模型训练，生成词向量
#    sentences = w2v.Text8Corpus("../datasets/text8")
#    model = w2v.Word2Vec(sentences, size=20, min_count=5)
    if sentences is None:
        model = w2v.load(model_dir)
    else:
        model = w2v.Word2Vec(sentences, size=20, min_count=5)
        model.save(model_dir)
    '''
    sentences = w2v.LineSentence('trainword.txt')
    model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
    model.save(model_file_name)
    '''
    #model = w2v.Word2Vec.load(model_file_name)
    return model;

train_dir = "./datasets/data.csv"

#texts,labels,labels_index,index_lables=datahelper(train_dir)
texts,labels=datahelper(train_dir)
#textCNN模型
print("data done..")
class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        vocb_size = args['vocb_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix=args['embedding_matrix']
        #需要将事先训练好的词向量载入
        self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2),

                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,64,64)
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2)
                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.embeding(x)
        x=x.view(x.size(0),1,max_len,word_dim)
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        #print(x.size())
        output = self.out(x)
        return output
#词表
word_vocb=[]
word_vocb.append('')
for text in texts:
    for word in text:
        word_vocb.append(word)
word_vocb=set(word_vocb)
vocb_size=len(word_vocb)
#设置词表大小
nb_words=40000
max_len=64;
word_dim=parser.word_dim
n_class=2

args={}
if nb_words<vocb_size:
    nb_words=vocb_size;
#textCNN调用的参数
args['vocb_size']=nb_words
args['max_len']=max_len
args['n_class']=n_class
args['dim']=word_dim

EPOCH=1000

texts_with_id=np.zeros([len(texts),max_len])
#词表与索引的map
word_to_idx={word:i for i,word in enumerate(word_vocb)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}
#每个单词的对应的词向量
if args.use_self:
    embeddings_index = getw2v(args.w2v)
else:
    embeddings_index = getw2v(args.w2v, texts)
#预先处理好的词向量
embedding_matrix = np.zeros((nb_words, word_dim))
random_dict = {}
for word, i in word_to_idx.items():
    if i >= nb_words:
        continue
    if parser.use_random or parser.use_normal:
        if word not in random_dict:
            if parser.use_random:
                random_dict[word] = np.random.uniform(-1,1,size=[word_dim])
            else:
                random_dict[word] = np.random.normal(size=[word_dim])
        embedding_matrix[i] = random_dict[word]
    else:
        if word in embeddings_index:
            embedding_vector = embeddings_index[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
args['embedding_matrix']=torch.Tensor(embedding_matrix)
#构建textCNN模型
print("Word embedding done..")
cnn=textCNN(args).to(device)

#生成训练数据，需要将训练数据的Word转换为word的索引
for i in range(0,len(texts)):
    if len(texts[i])<max_len:
        for j in range(0,len(texts[i])):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]
        for j in range(len(texts[i]),max_len):
            texts_with_id[i][j] = word_to_idx['']
    else:
        for j in range(0,max_len):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]

LR = 0.00001
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#损失函数
loss_function = nn.CrossEntropyLoss()
#训练批次大小
epoch_size=1000
texts_len=len(texts_with_id)
print(texts_len)
#划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(texts_with_id, labels, test_size=0.2, random_state=42)

#print(texts_with_id[2][3].type)
#print(y_test)
test_x=torch.LongTensor(x_test)
test_y=torch.LongTensor(y_test)
train_x=x_train
train_y=y_train

test_epoch_size=300;
f = open(model_dir+"acc.log", "w")
best_acc = 0
for epoch in range(EPOCH):

    #Train
    for i in range(0,(int)(len(train_x)/epoch_size)):

        b_x = Variable(torch.LongTensor(train_x[i*epoch_size:i*epoch_size+epoch_size]))
        b_y = Variable(torch.LongTensor((train_y[i*epoch_size:i*epoch_size+epoch_size])))
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        print(str(i), loss)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (b_y == pred_y)
        acc = acc.numpy().sum()
        accuracy = acc / (b_y.size(0))

    #Evalue
    acc_all = 0
    sum_all = 0
    with torch.no_grad():
        for j in range(0, (int)(len(test_x) / test_epoch_size)):
            b_x = Variable(torch.LongTensor(test_x[j * test_epoch_size:j * test_epoch_size + test_epoch_size]))
            b_y = Variable(torch.LongTensor((test_y[j * test_epoch_size:j * test_epoch_size + test_epoch_size])))
            b_x, b_y = b_x.to(device), b_y.to(device)
            test_output = cnn(b_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            sum_all = j * test_epoch_size + test_epoch_size
            acc = (pred_y == b_y)
            acc = acc.numpy().sum()
#            print("acc " + str(acc / b_y.size(0)))
            acc_all = acc_all + acc

    accuracy = acc_all / (sum_all)
    if accuracy > best_acc:
        with open(model_dir+"best_acc.txt", "w") as f3:
            f3.write("EPOCH %d, best_acc %.7f%" % (epoch + 1, acc))
        best_acc = accuracy
        #save model
        print("Best model: saving...")
        
    #Output result
    result = "EPOCH %d, acc %.7f%" % (epoch + 1, accuracy)
    print(result)
    f.write(result+"\n")
    f.flush()
    
    #Saving model
    if epoch % parser.save_every == 0:
        print('Saving model...')
        torch.save(net.state_dict(), '%snet_%03d.pth' % (model_dir, epoch + 1))
f.close()
