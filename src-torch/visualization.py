##################################################################
# AIM OF THE SCRIPT                                              #
##################################################################
# read through checkpoints/                                      #
# get params from dir names                                      #
# get stats from each subfolder                                  #
# generate scattered plot of scores in param space               #
##################################################################

##################################################################
# IMPORTS SECTIONS                                               #
##################################################################
import argparse                                                  #
import numpy as np                                               #
import pandas as pd                                              #
import os.path as path                                           #
import os                                                        #
import subprocess                                                #
import matplotlib.pyplot as plt                                  #
##################################################################

def slowlyReadlines(fnames):
   with open(fname, 'rb') as f:
      print('start reading')
      first = f.readline()    # Read the first line
      f.seek(-2, 2)             # Jump to the second last byte.
      print('read first line')
      while f.read(1) != b"\n":# Until EOL is found...
         f.seek(-2, 1)         # ...jump back the read byte plus one more.
         last = f.readline()   # Read the last line
   return first, last 

def readlines(fname):
    first = subprocess.check_output(['head', '-1', fname]) 
    last = subprocess.check_output(['tail', '-1', fname]) 
    return first, last

def tonumber(value):
   try:
      return float(value)
   except:
      return value

def replaceNan(df,cols, vals):
   if type(cols) == list and type(vals) == list and len(cols) != len(vals):
      return
   df.fillna({col: val for col, val in zip(cols,vals)}, axis=0, inplace=True)
   return


def plotScattered(x,y, results, fig, xlabel, ylabel):
   x_scatter = np.asarray(x, dtype=np.float64)
   y_scatter = np.asarray(y, dtype=np.float64)
   axs = []
   im = []
   marker_size = 100

   # Set common labels
   fig.text(0.5, 0.04, xlabel, ha='center', va='center')
   fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical')

   try:
      for i in xrange(results.shape[-1]):
         colors = np.asarray(results.iloc[:,i])
         axs.append(fig.add_subplot(results.shape[-1], 1, i+1))
         im.append(axs[i].scatter(x_scatter, y_scatter, marker_size,c=colors))
         fig.colorbar(im[i],ax=axs[i])
         # axs[i].set_title(results.column[i])
         # ax[i+1].set_title('ax1 title')

   # if results on a single array/series   
   except:
      colors = results
      axs.append(fig.add_subplot(1, 1, i+1))
      im.append(axs[i].scatter(x_scatter, y_scatter, marker_size))

   return axs

# create pandas daframe from experimental data
def getData(dataDir):

   experiments = []
   for (dirpath, dirnames, filenames) in os.walk(dataDir):
      if (dirnames is None) or (not filenames):
         continue

      d = {} 
      paramsList = str.split(str.split(dirpath,'/')[-2], ',')
      d = dict(map(lambda x: str.split(x, '='), filter(lambda x: '=' in x , paramsList)))

      logs = list(filter(lambda x: x == 'training.log' or x == 'loss.log' ,filenames))
      resultLines = [readlines(path.join(dirpath, log)) for log in logs]
      resultLines = list(filter(lambda x: x[0] != x[1], resultLines))
      for resultLine in resultLines:
         d.update( {str(el): tonumber(val) for el,val in zip(str.split(resultLine[0].rstrip(),'\t'), str.split(resultLine[1].rstrip(), '\t'))} )
 
      d['path'] = dirpath
      experiments.append(d)

   return pd.DataFrame(experiments)

def main():
   parser = argparse.ArgumentParser(description='Creates scattered plots of hyparams vs scores')
   parser.add_argument('--dir', help='Specifies the dir to look into')
   args = parser.parse_args()

   data = getData(args.dir)
    
   # default config values dont show up on the files
   replaceNan(data,['LR','weightDecay','depth'],[0.1, 1e-4, 34])

   # filter useful columns only
   colsPerEpoch = ['Training Acc Error','Testing Acc Error','Training Recall Error','Testing Recall Error','Training F1 Error','Validation F1 Error', 'Training Loss', 'Validation Loss']
   colsPerIter = ['Loss']
   colsParam = ['LR', 'weightDecay', 'Epoch', 'dataset']
   # colsParam = ['LR', 'depth', 'weightDecay', 'Epoch', 'dataset']
   data = (data[colsPerEpoch + colsPerIter + colsParam]).dropna(axis=0)

   # filter out unwanted datasets) 
   data = data[ data['dataset'] == 'cdnet' ].reset_index()
   data = data[ data['Epoch'] == 1 ].reset_index()
   
   #plot every train/test error pair (acc/recall/f1/loss) scattered in log(LR) vs log(weightDecay)
   #plot every train/test error pair vs depth
   x_scatter = np.log10(data.loc[:,'LR'].astype(np.float64))
   y_scatter = np.log10(data['weightDecay'].astype(np.float64))
   
   for col in xrange(0,len(colsPerEpoch),2):
      fig = plt.figure(1)
      dataName = colsPerEpoch[col].split(' ')
      # use val and train errors as color intensity in scattered plot
      colors = data.loc[:, [colsPerEpoch[col],colsPerEpoch[col+1]] ].astype(np.float64)
      plotScattered(x_scatter, y_scatter, colors, fig, 'log learning rate', 'log reg strength')
      fig.suptitle('CDNet ' + dataName[1], fontsize=14)
      # plt.show()
      plt.savefig('_'.join(dataName[1:]) + '-param_space.eps', format='eps', dpi=1000)
      plt.savefig('_'.join(dataName[1:]) + '-param_space.png', format='png')
      plt.clf()
   # fig = plt.figure()
   # plt.plot(np.arange(len(data.loc[:, 'Loss'])),data.loc[:, 'Loss'].astype(np.float64))
   # plt.show()

if __name__ == '__main__':
    main()
