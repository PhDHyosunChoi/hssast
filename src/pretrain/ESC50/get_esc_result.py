# -*- coding: utf-8 -*-
# @Time    : 11/15/20 1:04 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_esc_result.py

# summarize esc 5-fold cross validation result

import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")

if __name__ == '__main__':
    args = parser.parse_args()
    mAP_list = []
    acc_list = []
    for fold in range(1, 3): #[Hyosun] #[Original] 6):
        result = np.loadtxt(args.exp_path+'/fold' + str(fold) + '/result.csv', delimiter=',')
        #[Hyosun] added for check
        print("[Hyosun]result.shape: ", result.shape)
        print("[Hyosun]result.ndim : ", result.ndim)
        #print("[Hyosun][result.shape[0], result.shape[1]]: ", [result.shape[0], result.shape[1]]) 
        #[/Hyosun] added for check

        #[Hyosun] added
        if result.ndim == 1: #[Hyosun] added
          result = result.reshape(1, -1) #[Hyosun] convert into row vector 2d array
          print("[Hyosun:after reshape]result.shape: ", result.shape)
          print("[Hyosun:after reshape]result.ndim : ", result.ndim)
        #[/Hyosun] added
        
        if fold == 1:
            # #[Hyosun] edited
            # if result.ndim == 1: #[Hyosun] added
              # result.reshape(1, -1)
              # cum_result = np.zeros(result.shape[0])
            # else: #[Hyosun] the original code
          print("[Hyosun][result.shape[0], result.shape[1]]: ", [result.shape[0], result.shape[1]])
          cum_result = np.zeros([result.shape[0], result.shape[1]])
            # # [/Hyosun] edited
        cum_result = cum_result + result
    result = cum_result / 5 #[Hyosun] now result = average result, with the same shape as before
    # #[Hyosun] added
    # if result.ndim == 1:
    #   result.reshape(1, -1)
    # #[/Hyosun] added
    np.savetxt(args.exp_path+'/result.csv', result, delimiter=',')
    # note this is choose the best epoch based on AVERAGED accuracy across 5 folds, not the best epoch for each fold
    ## [Hyosun] edited
    # if result.ndim == 1: #[Hyosun] added 
    #   best_epoch = np.argmax(result[:])
    #   np.savetxt(args.exp_path + '/best_result.csv', result[best_epoch], delimiter=',')
    # else: #[Hyosun] the original code
    best_epoch = np.argmax(result[:, 0]) #[Hyosun] 0번째 열 원소들이 무엇을 의미하는가?(바로 아래 comment참고)
    # [Hyosun] added: result.append([train_acc_meter.avg, train_nce_meter.avg, acc_eval, nce_eval, optimizer.param_groups[0]['lr']])
    np.savetxt(args.exp_path + '/best_result.csv', result[best_epoch, :], delimiter=',')
    ##[/Hyosun] editied


    acc_fold = []
    print('--------------Result Summary--------------')
    for fold in range(1, 3): #[Hyosun] #[Original] 6):
        result = np.loadtxt(args.exp_path+'/fold' + str(fold) + '/result.csv', delimiter=',')
        # note this is the best epoch based on AVERAGED accuracy across 5 folds, not the best epoch for each fold (which leads to over-optimistic results), this gives more fair result.
        
        #[Hyosun] added for check
        print("[Hyosun]result.shape: ", result.shape)
        print("[Hyosun]result.ndim : ", result.ndim)
        #print("[Hyosun][result.shape[0], result.shape[1]]: ", [result.shape[0], result.shape[1]]) 
        #[/Hyosun] added for check

        #[Hyosun] added
        if result.ndim == 1: #[Hyosun] added
          result = result.reshape(1, -1) #[Hyosun] convert into row vector 2d array
          print("[Hyosun:after reshape]result.shape: ", result.shape)
          print("[Hyosun:after reshape]result.ndim : ", result.ndim)
        #[/Hyosun] added
                      
        # #[Hyosun] edited
        # if result.ndim == 1: #[Hyosun] added 
        #   acc_fold.append(result[best_epoch])
        #   print('Fold {:d} accuracy: {:.4f}'.format(fold, result[best_epoch]))
        # else: #[Hyosun] the original code
        acc_fold.append(result[best_epoch, 0]) #[Hyosun] 0번째 열 원소들이 무엇을 의미하는가? 
        print('Fold {:d} accuracy: {:.4f}'.format(fold, result[best_epoch, 0]))
        # #[/Hyosun] editied
    acc_fold.append(np.mean(acc_fold)) #[Hyosun] append the mean(acc_fold) as the last element
    print('The averaged accuracy of 5 folds is {:.3f}'.format(acc_fold[-1]))
    np.savetxt(args.exp_path + '/acc_fold.csv', acc_fold, delimiter=',')
