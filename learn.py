#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse,glob,os,cPickle,numpy as np,time,learnC as lc,bottleneck as bn

#Parse feature file to generate a list
def featureFileNoLabelToList(filename):
    f=open(filename,'r')
    content = [line.rstrip() for line in f]
    featureList=[]
    for lines in content:
        tokens=lines.split(' ')
        featureDict={}
        for feature in tokens:
            kv=feature.split(':')
            #print kv
            featureDict[int(kv[0])]=int(kv[1])
        featureList.append(featureDict)
        #print featureList
    return featureList



#normalize 1d array to sum=1
def normalize1(a):
    s=np.sum(a)
    if s!=0:
        normed=np.divide(a,s)
    return normed

#nomalize 2d arry to col sum=1
# def normalize2(a):
#     normed=np.empty_like(a)
#     #at=a.T
#     #normedT=np.empty_like(at)
#     for i in range(len(a)):
#         normed[i]=normalize1(a[i])

#     return normed

def normalize2(a):
    #normed=np.empty_like(a)
    at=a.T
    normedT=np.empty_like(at)
    for i in range(len(at)):
        normedT[i]=normalize1(at[i])
    normed=normedT.T
    return normed

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

#plsi class that contains plsi routines and p values
class plsi(object):
    def __init__(self):
        self.p_dz_n=None
        self.p_wz_n=None
        self.p_z_n=None
        self.log_like=0
        self.time=0
        #self.new_log_like=0

        #initialize p arrays
        #@timing
    def initialize(self,d,w,z):
        self.p_dz_n=normalize2(np.random.rand(d,z))
        self.p_wz_n=normalize2(np.random.rand(w,z))
        self.p_z_n=normalize1(np.random.rand(z))
        # self.p_dz_n=np.random.rand(d,z)
        # self.p_wz_n=(np.random.rand(w,z))
        # self.p_z_n=(np.random.rand(z))
        # self.numerator_p_dz_n=np.zeros((d,z),dtype=np.float64)
        # self.denominator_p_dz_n=np.zeros(z,dtype=np.float64)
        # self.numerator_p_wz_n=np.zeros((w,z),dtype=np.float64)
        # self.denominator_p_wz_n=np.zeros(z,dtype=np.float64)
        # self.numerator_p_z_n=np.zeros(z,dtype=np.float64)
        # self.denominator_p_z_n=np.zeros(z,dtype=np.float64)
        self.time=0
        self.log_like=0
        #print self.p_z_n
        #print self.p_dz_n

        #@timing
    def train(self,featureList,docNum,wordNum,z):
        self.initialize(docNum,wordNum,z)
        #new_log_like=self.loglikelihood(featureList,docNum,wordNum,z)
        #self.log_like=new_log_like
        #cycle=0
        time1 = time.time()
        while True:
            self.update(featureList,docNum,wordNum,z)
            new_log_like=self.loglikelihood(featureList,docNum,wordNum,z)#self.loglikelihood(featureList,docNum,wordNum,z)
            print "old log_like:"+str(self.log_like)
            print "new log_like:"+str(new_log_like)
            delta=new_log_like-self.log_like
            print "delta:"+str(delta)
            self.log_like=new_log_like
            if np.abs(delta)<1:
                break
            #cycle+=1
        time2 = time.time()
        self.time=(time2-time1)

            #@timing
    def update(self,featureList,docNum,wordNum,z):

        #call C++ routine for speed
        res=lc.update(featureList,docNum,wordNum,z,self.p_wz_n.tolist(),self.p_dz_n.tolist(),self.p_z_n.tolist())
        self.p_wz_n=np.array(res[0]);
        self.p_dz_n=np.array(res[1]);
        self.p_z_n=np.array(res[2]);
        # print "npsums:\n"
        # print np.sum(self.p_wz_n,axis=0)
        # print np.sum(self.p_dz_n,axis=0)
        # print np.sum(self.p_z_n)
        #print res
        #update
        # for d in range(docNum):
        #     for w in featureList[d]:
        #         w=w-1
        #         denominator=0
        #         #equivalent vector expression:
        #         numerator=self.p_dz_n[d][:]*self.p_wz_n[w][:]*self.p_z_n[:]
        #         denominator=np.sum(numerator)
        #         P_z_condition_d_w=numerator/denominator
        #         tfwd=featureList[d][w+1]
        #         self.numerator_p_dz_n[d][:]+=tfwd*P_z_condition_d_w[:]
        #         self.denominator_p_dz_n[:]+=tfwd*P_z_condition_d_w[:]
        #         self.numerator_p_wz_n[w][:]+=tfwd*P_z_condition_d_w[:]
        #         self.denominator_p_wz_n[:]+=tfwd*P_z_condition_d_w[:]
        #         self.numerator_p_z_n[:]+=tfwd*P_z_condition_d_w[:]
        #         self.denominator_p_z_n[:]+=tfwd

        # for d in range(docNum):
        #     self.p_dz_n[d][:]=self.numerator_p_dz_n[d][:]/self.denominator_p_dz_n[:]


        # for w in range(wordNum):
        #     self.p_wz_n[w][:]=self.numerator_p_wz_n[w][:]/self.denominator_p_wz_n[:]


        # self.p_z_n[:]=self.numerator_p_z_n[:]/self.denominator_p_z_n[:]


        #@timing
    def loglikelihood(self,featureList,docNum,wordNum,z):
        #call C++ routine for speed
        new_log_like=lc.loglikelihood(featureList,docNum,wordNum,z,self.p_wz_n.tolist(),self.p_dz_n.tolist(),self.p_z_n.tolist())

        # original numpy implementation, a bit slow
        # new_log_like=0
        # for d in range(docNum):
        #     for w in featureList[d]:
        #         w=w-1
        #         tfwd=featureList[d][w+1]
        #         p_d_w=np.sum(self.p_wz_n[w][:]*self.p_dz_n[d][:]*self.p_z_n[:])
        #         new_log_like+=tfwd*np.log(p_d_w)
        return new_log_like

    def output(self,outpath,words,z):
        #topIndices=bn.argpartsort(-self.p_wz_n[:][z])
        with open(os.path.join(outpath,str(z)+"-topic.txt"),'w') as outfile:
            for i in range(z):
                #print self.p_wz_n.shape
                #print self.p_wz_n[:,i]
                topIndices=bn.argpartsort(-self.p_wz_n[:,i],20)[:20]
                #                print topIndices
                topList=[]
                for index in topIndices:
                    topList.append([index,self.p_wz_n[index,i]])
                sortedList=sorted(topList,key=lambda x:(-x[1]))

                outfile.write("Topic "+str(i)+":\n")
                for w in sortedList:
                    outfile.write(words[w[0]+1]+":"+str(w[1])+"\n")


        with open(os.path.join(outpath,"k-likelihood-time.txt"),'a') as outfile:
            outfile.write(str(z)+' '+str(self.log_like)+' '+str(self.time)+'\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser("Learn PLSI")
    parser.add_argument('-f','--feature_file',help='Input feature file path',required=True)
    parser.add_argument('-o','--output_folder',help='Output file path',required=True)
    parser.add_argument('-w','--word_file',help='Word file name',required=True)
    parser.add_argument('-k1','--k_min',help='Topic number k min',required=True)
    parser.add_argument('-k2','--k_max',help='Topic number k max',required=True)
    parser.add_argument('-ks','--k_step',help='Topic number k incremental step',required=True)
    args=parser.parse_args()

    featureList=featureFileNoLabelToList(args.feature_file)
    docNum=len(featureList)
    print "doc number:"+str(docNum)

    #open word index file
    wordIndex=cPickle.load(open(args.word_file,'rb'))
    #swap wordIndex dict to generate index-word dict
    words=dict((wordIndex[k][0],k) for k in wordIndex)
    #print words
    wordNum=len(words)
    print "word number:"+str(wordNum)

    if not os.path.exists(args.output_folder):
        try:
            os.makedirs(args.output_folder)
        except OSError,why:
            print "Failed: %s"%str(why)
    if os.path.exists(os.path.join(args.output_folder,"k-likelihood-time.txt")):
        os.remove(os.path.join(args.output_folder,"k-likelihood-time.txt"))

    for k in range(int(args.k_min),int(args.k_max),int(args.k_step)):
        plsiOb=plsi()
        plsiOb.train(featureList,docNum,wordNum,k)
        plsiOb.output(args.output_folder,words,k)
