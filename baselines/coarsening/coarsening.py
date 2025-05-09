import numpy as np

class solver_v2:
  def __init__(self, X, L, k, lambda_param, beta_param, alpha_param, gamma_param):
    self.X = X
    self.L = L
    self.p = X.shape[0]
    self.k = k
    self.n = X.shape[1]

    n = self.n
    k = self.k
    p = self.p

    self.thresh = 1e-10 # The 0-level
    self.X_tilde = np.random.normal(0, 1, (k, n))
    self.C = np.random.normal(0,1,(p,k))
    self.C[self.C < self.thresh] = self.thresh

    self.w = np.random.normal(10, 1, (k*(k-1))//2)
    self.w[self.w < self.thresh] = self.thresh

    self.beta_param = beta_param
    self.alpha_param = alpha_param
    self.lambda_param = lambda_param
    self.gamma_param = gamma_param
    self.iters = 0
    self.lr0 = 1e-5

  def getLR(self):
    a = 0.99
    return self.lr0

  def calc_f(self):
    X_tilde = self.X_tilde
    fw = 0
    fw += np.trace(X_tilde.T@self.C.T@self.L@self.C @X_tilde)
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    fw -= self.gamma_param*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1]
    fw += (self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2
    fw += (self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2)
    return fw

  def update_X_tilde(self):
    L_tilde = self.C.T@self.L@self.C
    A = 2*L_tilde/(self.alpha_param)
    A = A + np.dot(self.C.T, self.C)
    b = np.dot(self.C.T, self.X)
    self.X_tilde = np.linalg.pinv(A)@b

    for i in range(len(self.X_tilde)):
      self.X_tilde[i] = (self.X_tilde[i]/(np.linalg.norm(self.X_tilde[i])))
    return None

  def grad_C(self):
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    v=np.linalg.pinv(self.C.T@self.L@self.C + J)
    gradC = np.zeros(self.C.shape)
    gradC += self.alpha_param*((self.C@self.X_tilde - self.X)@self.X_tilde.T)
    gradC += (self.lambda_param) * (np.abs(self.C) @ (np.ones((self.k, self.k))))
    gradC += -2*(self.gamma_param)*self.L@self.C@v
    gradC += 2*self.L@self.C@self.X_tilde@self.X_tilde.T
    return gradC

  def update_C(self, lr = None):
    if not lr:
      lr = 1/ (self.k)
    lr = self.getLR()
    C = self.C
    C = C - lr*self.grad_C()
    C[C<self.thresh] = self.thresh
    self.C = C
    C = self.C.copy()

    for i in range(len(C)):
      C[i] = C[i]/np.linalg.norm(C[i],1)

    self.C = C.copy()
    return None


  def fit(self, max_iters):
    ls = []
    MAX_ITER_INT = 100
    for i in range(max_iters):
      for _ in range(MAX_ITER_INT):
        self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
    return (self.C, self.X_tilde, ls )

  def New_fit(self):
    ls=[]
    MAX_ITER_INT = 100
    while(True):
      C_prev=self.C
      self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      if(np.linalg.norm(self.C-C_prev)<0.1):
          return (self.C, self.X_tilde, ls )
    return (self.C, self.X_tilde, ls )

  def set_experiment(self, X, X_t):
    self.X = X
    self.X_tilde = X_t

