import numpy as np
import scipy as sp
import scipy.linalg as sl

def ransac(date,model,n,k,t,d,debug = False,return_all = False):
    iterations = 0
    bestFit = None
    bestErr = np.inf
    best_inlier_idxs = None
    while iterations<k:
        maybe_idxs,test_idxs = random_partition(n,date.shape[0])
        print('test_idxs=',test_idxs)
        maybe_inliers = date[maybe_idxs,:]
        test_points = date[test_idxs]
        maybemobel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points,maybemobel)
        print('test_err',test_err<t)
        also_idxs = test_idxs[test_err<t]
        print('also_idxs=',also_idxs)
        also_inliers = date[also_idxs,:]
        print('d=',d)
        if (len(also_inliers)>d):
            bestDate = np.concatenate((maybe_inliers,also_inliers))
            bestmodel = model.fit(bestDate)
            best_errs = model.get_error(bestDate,bestmodel)
            thisErr = np.mean(best_errs)
            if  thisErr<bestErr:
                bestFit = bestmodel
                bestErr = thisErr
                best_inlier_idxs = np.concatenate((maybe_idxs,also_idxs))
        iterations +=1
    if bestFit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestFit,{'inliers':best_inlier_idxs}
    else:
        return bestFit
def random_partition(n,n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1,idxs2
class LinearLeastSquareModel:
    def __init__(self,input_columns,output_columns,debug = False):
        self.input_colimns = input_columns
        self.output_colimns = output_columns
        self.debug = debug

    def fit(self,data):
        A = np.vstack([data[:,i] for i in self.input_colimns]).T
        B = np.vstack([data[:,i] for i in self.output_colimns]).T
        x,resids,rank,s = sl.lstsq(A,B)
        return x
    def get_error(self,data,model):
        A = np.vstack([data[:,i] for i in self.input_colimns]).T
        B = np.vstack([data[:, i] for i in self.output_colimns]).T
        B_fit = sp.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1)
        return err_per_point
def test():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*np.random.random((n_samples,n_inputs))
    perfect_fit = 60*np.random.normal(size = (n_inputs,n_outputs))
    B_exact = sp.dot(A_exact,perfect_fit)
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)
    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20*np.random.random((n_outliers,n_inputs))
        B_noisy[outlier_idxs] = 50*np.random.normal(size=(n_outliers,n_outputs))
        all_data = np.hstack((A_noisy,B_noisy))
        input_colimns = range(n_inputs)
        output_colimns = [n_inputs+i for i in range(n_outputs)]
        debug = False
        model = LinearLeastSquareModel(input_colimns,output_colimns,debug=debug)
        linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_colimns],all_data[:,output_colimns])
        ransac_fit,ransac_data = ransac(all_data,model,50,1000,7e3,300,debug=debug,return_all=True)

    if 1:
        import pylab
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs]
        if 1:
            pylab.plot(A_noisy[:,0],B_noisy[:,0],'k.',label = 'data')
            pylab.plot(A_noisy[ransac_data['inliers'],0],B_noisy[ransac_data['inliers'],0],'bx',label = 'RANSAC data')
        else:
            pylab.plot(A_noisy[non_outlier_idxs,0],B_noisy[non_outlier_idxs,0],'k.',label = 'noisy data')
            pylab.plot(A_noisy[outlier,0],B_noisy[outlier_idxs,0],'r.',label='outlier data')

        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted,ransac_fit[:,0]),label = 'RANSAC fit')
        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted,perfect_fit[:,0]),label = 'exact system')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit[:, 0]), label='liner fit')
        pylab.legend()
        pylab.show()
if __name__=='__main__':
    test()
