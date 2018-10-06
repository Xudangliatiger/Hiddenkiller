import Bio as bio
from Bio import SeqIO
import pandas as pd
data=SeqIO.parse("anonymous_gsa (1).fasta", "fasta")
for seq_record in SeqIO.parse("anonymous_gsa (1).fasta", "fasta"):
    f=open('seq100x.txt','a') //写入txt
    a=''
    length=len(seq_record.seq) //seq长度
    xx=int(length/100) //每一百个碱基分一段
    id=str(seq_record.id)
    for i in range(0,xx-1):
        a=a+id+','+str(seq_record.seq[i*100:100*(i+1)-1])+'\n'       
    a=a+id+','+str(seq_record.seq[xx*100:l])+'\n'
    f.write(a)
    f.close()