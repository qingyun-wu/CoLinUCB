'''
Created on May 21, 2015

@author: hongning
'''
import graphlab as gl;
import sys

def LoadData(filename):
    collection = []
    with open(filename) as reader:
        for line in reader:
            elm = []
            for x in line.split():
                elm += [float(x)]
            collection += [elm]
    print '%d instances loaded from %s' % (len(collection), filename)
    return collection

def SaveModel(filename, model):
    with open(filename, 'w') as writer:
        for centriod in model['cluster_info']['X1']:
            for x in centriod:
                writer.write(str(x) + ' '),
            writer.write('\n')
        

if __name__ == '__main__':
    if len(sys.argv) == 3:
        k = int(sys.argv[1])
        dataset = gl.SFrame(gl.SArray(LoadData(sys.argv[2])))
        model = gl.kmeans.create(dataset, num_clusters=k, max_iterations=100)
        model['cluster_info']
        
        SaveModel('kmeans_model.dat', model)