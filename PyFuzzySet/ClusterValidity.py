#@title <b>Cluster Validity Indexes</b>

import numpy as np
class Indexes:
    """
    Cluster Validity Index (Küme Geçerlilik İndeksi)
    """

    def __init__(self, u, v, m, data):
        self.u=u
        self.v=v
        self.m=m
        self.data=data

    def PC(self):
        """
        Partition Coefficient (Bölünme Kat Sayısı)    
        """
        return (self.u**2).sum()/len(self.data)
    
    def CE(self):
        """
        Classification Entropy (Sınıflandırma Entropisi)
        """
        return (self.u*np.log(self.u)).sum()/-len(self.data)
    
    def MPC(self):
        """
        Modificated Partititon Coefficient (Gözden geçirilmiş Bölünme Katsayısı)\n
        Normalized Partititon Coefficient (Normalleştirilmiş Bölünme Katsayısı)
        """
        c=self.u.shape[1]
        return 1-(c/(c-1))*(1-self.PC())
    
    def XB(self):
        """
        Xie-Beni Index kodları buraya gelecek
        """
        pass