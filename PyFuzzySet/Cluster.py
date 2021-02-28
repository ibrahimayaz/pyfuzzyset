#@title <b>Fuzzy Sets and Intuitionistic Fuzzy Sets Algorithm</b>

import numpy as np

class FCM:
    """
    Fuzzy C Means (Bulanık C Ortalamalar)
    """
    def __init__(self, c, m, eps, maxIter):
        """
        Parameters
        ----------
        c : int
            Number of Cluster (Küme Sayısı)
        m : float
            Fuzziness (Bulanıklaştırıcı) 1<=m
        eps : float
            Epsilon (Durdurma Kriteri) 0<eps<1
        maxIter : int
            Maximum number of iterations (Maksimum iterasyon sayısı)

        -------

        """
        self.c = c
        self.m = m
        self.eps = eps
        self.maxIter = maxIter

    def InitialMembership_Creation(self, data):
        U = np.random.randint(100, 200, size=(len(data), self.c))
        return U/U.sum(axis=1, keepdims=True)

    def V(self, U, data):
        U_m = U**self.m
        U_m = U_m/U_m.sum(axis=0, keepdims=True)
        return np.matmul(U_m.T, data)

    def U(self, V_old, data):
        U_new = np.zeros(shape=(data.shape[0], self.c))
        for i in range(U_new.shape[0]):
            for j in range(U_new.shape[1]):
                U_new[i][j] = 1./np.linalg.norm(data[i]-V_old[j])

        U_new = U_new**(2./(self.m-1))
        U_new = U_new/U_new.sum(axis=1, keepdims=True)
        return U_new

    def SSE(self, U, V, data):
        U_m = U**self.m
        result = 0
        for j in range(self.c):
            for i in range(data.shape[0]):
                result += U_m[i][j] * np.linalg.norm(data[i]-V[j])**2
        return result

    def Cluster(self, data):
        U = self.InitialMembership_Creation(data)
        print("INFO: The initial membership matrix has been created.")
        V = np.zeros((self.c, data.shape[0]))
        error_old = float(self.SSE(U, V, data))
        iteration = 1
        loss_values = []
        while iteration <= self.maxIter:
            V = self.V(U, data)
            U = self.U(V, data)
            error_new = float(self.SSE(U, V, data))
            loss = error_old - error_new
            loss_values.append(loss)
            if loss < self.eps:
                break
            iteration += 1
            error_old = error_new
        print("INFO: Clustering Complete ! ")
        return U, V, loss_values

    def J(self, U, V, data):
        """
        Objective Function (Amaç Fonksiyonu)
        """
        result=0
        for i in range(len(U)):
            for j in range(self.c):
                result+=(U[i,j]**self.m)*self.Dist(data[i]-V[j])
        return result

    def Dist(self, V, data):
        return np.sqrt((data-V)**2)

class IFCM:
    """
    Intuitionistic Fuzzy C Means (Sezgisel Bulanık C Ortalamalar)
    """

    def __init__(self, c, m, eps, maxIter, lam=2):
        """
          Parameters
          ----------
          c : int
              Number of Cluster (Küme Sayısı)
          m : float
              Fuzzification Value (Bulanıklaştırıcı Değeri) 1<=m
          eps : float
              Epsilon (Durdurma Kriteri) 0<eps<1
          maxIter : int
              Maximum number of iterations (Maksimum iterasyon sayısı)
          lam : float
              The lambda parameter value in Z's negation(non-membership) function
              Z'nin üye olmama işlevindeki lambda parametre değeri
              Default value:2

          -------

        """
        self.c = c
        self.m = m
        self.eps = eps
        self.maxIter = maxIter
        self.lam = lam

    def Cluster(self, data):
        U = self.InitialMembership_Creation(data)
        print("INFO: The initial membership matrix has been created.")
        V = [(0,0,0) for i in range(self.c)]
        iteration=1
        Z,u_z,n_z,p_z=self.Z(data)
        loss_values=[]
        while iteration<=self.maxIter:          
            v_old=V
            V=self.V(U, u_z, n_z, p_z)
            U=self.U(Z, V, data)
            v_new=V
            loss=self.Error_Calc(v_old, v_new)
            loss_values.append(loss)
            if (loss<self.eps):
                break
            iteration+=1
        print("INFO: Clustering Complete ! ")
        return U, V, Z, loss_values

    def InitialMembership_Creation(self, data):
        """
        Initial Membership Creation (Başlangıç Üyelik Matrisinin Oluşturulması)
        Üyelik matrisi rastgele oluşturulur. Oluşturulan matrisin her satırının toplamı 1 olacak şekilde ayarlanır.
        """
        U = np.random.randint(100, 200, size=(len(data), self.c))
        return U/U.sum(axis=1, keepdims=True)

    def U(self, Z, V, data):
        """
        Membership Matrix (Üyelik Matrisi)
        """
        U_yeni = np.zeros(shape=(data.shape[0], self.c))
        for i in range(U_yeni.shape[0]):
            for j in range(U_yeni.shape[1]):
                U_yeni[i][j] = 1./self.Uzaklik(Z[i], V[j])

        U_yeni = U_yeni**(2./(self.m-1))
        U_yeni = U_yeni/U_yeni.sum(axis=1, keepdims=True)
        return U_yeni

    def V(self, U, u_z, n_z, p_z):
        """
        Cluster Center (Küme Merkezi)
        """
        U_m = U**self.m
        U_m = U_m/U_m.sum(axis=0, keepdims=True)
        u_v=np.matmul(U_m.T, u_z)
        n_v=np.matmul(U_m.T, n_z)
        p_v=np.matmul(U_m.T, p_z)

        v=zip(u_v, n_v, p_v)

        return list(v)
        
    def Z(self, data):
        """
        General Z Matrix (Genel Z Matrisi)
        Includes the degrees of Z's membership, non-membership and fuzziness
        Z'nin üye olma, üye olmama ve belirsizlik derecelerini içerir
        """
        u=self.Z_Membership(data)
        n=self.Z_NonMembership(u)
        p=self.Z_Fuzziness(u,n)
        z=zip(u,n,p)
        return list(z), u, n, p

    def Z_Membership(self, data):
        """
        Z Membership Degree (Z'nin Üye Olma Derecesi)
        """
        max_value=np.max(data)
        min_value=np.min(data)
        u=[]
        for x in data:
            u.append((x-min_value)/(max_value-min_value))
        return u

    def Z_NonMembership(self, U_z):
        """
        Z Non-Membership Degree (Z'nin Üye Olmama Derecesi)
        """
        n=[]
        for u in U_z:
            n.append((1-u)/(1+self.lam*u))
        return n

    def Z_Fuzziness(self, U_z, N_z):
        """
        Z Fuzziness Degree (Z'nin Belirsizlik Derecesi)
        """
        pi=[]
        for i in range(len(U_z)):
            pi.append(1-(U_z[i]+N_z[i]))
        return pi

    def Error_Calc(self, V_old, V_new):
        result=0
        for i in range(self.c):
            result+=self.Dist(V_old[i], V_new[i])
        return result/self.c

    def Dist(self, Z, V, W=1/2):
        result=((Z[0]-V[0])**2 + (Z[1]-V[1])**2 + (Z[2]-V[2])**2)
        return W * np.sqrt(result)

    def J(self,U,V,Z):
        """
        Objective Function (Amaç Fonksiyonu)
        """
        result=0
        for j in range(len(Z)):
            for i in range(self.c):
                result+=(U[j,i]**self.m)*self.Dist(Z[j],V[i])
        return result