import numpy as np

def random_hamiltonian(size):
    
    """
    Create a random hamiltonian (h, J), the first is a unique dictionnary where all links are listed once, the second keeps all links
    """
    
    links = dict()
    for i in range (size):
        a = np.random.randint(0,size)
        w = np.unique(np.random.randint(i,size,a))
        j = np.random.standard_cauchy(a)
        links[i] = [(w[p],j[p]) for p in range (len(w)) if a > 0 ]
        
    all_links = dict()
    
    for k, v in links.items():
        c = v.copy()
        if k == 0 : 
            all_links[k] = c
        else : 
            for i in range (k-1, -1,-1):
                for j in links[i]:
                    if k == j[0]:
                        c.append((i, j[1]))
            all_links[k] = c    

    return links, all_links