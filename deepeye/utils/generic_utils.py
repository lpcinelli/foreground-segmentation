

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class History(AverageMeter):
    """ Stores values and computes some metrics """
    def reset(self):
        super(self, AverageMeter).reset()
        self.vals = []

    def update(self, val, n=1):
        super(self, AverageMeter).update(val, n)
        self.vals.append(val * n)

def find_threshold(y_pred, y_true, metric, min_val=0, max_val=1.0, eps=1e-6):
    '''
    Finds the best threshold considering a metric and a training set.

    @param y_pred Current predicted values to be thresholded.
    @param y_true Target labels.
    @param metric Metric to be maximized.
    @param min_val Minimum possible predicted value.
    @param max_val Maximum possible predicted value.
    @param eps Minimum interval

    @return Best threshold and metric value.
    '''

    # Set initial values
    thrs_cur = (min_val+max_val)/2.0
    thrs_low = min_val
    thrs_hgh = max_val

    # Bisection algorithm
    while(True):

        # Initializing bisection algorithm
        thrs_cur = (min_val+max_val)/2.0
        thrs_low = (thrs_cur+min_val)/2.0
        thrs_hgh = (thrs_cur+max_val)/2.0

        # Scores
        scr_low = metric(y_pred, y_true, thrs_low)
        scr_hgh = metric(y_pred, y_true, thrs_hgh)

        # Testing
        if (scr_low >= scr_hgh):
            # Update
            max_val = thrs_hgh
        else:
            min_val = thrs_low

        # End condition
        if (abs(max_val-min_val)<=eps):
            break

    # Return
    return thrs_cur, scr_low