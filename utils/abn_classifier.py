import chainer
import chainer.functions as F
import chainer.links as L

from chainer import reporter


class ABNClassifier(L.Classifier):

    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy):
        super(ABNClassifier, self).__init__(
            predictor, lossfun=lossfun, accfun=accfun)

    def __call__(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.ah = None
        self.y = None
        self.attention = None
        self.loss = None
        self.accuracy = None
        self.ah, self.y, self.attention = self.predictor(*args, **kwargs)
        if not chainer.config.train:
            self.loss = self.lossfun(self.y, t)
        else:
            att_loss = self.lossfun(self.ah, t)
            per_loss = self.lossfun(self.y, t)
            self.loss = att_loss + per_loss

        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss