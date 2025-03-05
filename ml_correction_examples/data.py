import json

sigma = [('Si',4.2), ('C',3.4)]
epsilon = [('C',0.0046), ('Si',0.0077)]
de = [('Si','C',4.35), ('Si',0.9862), ('C',2.423)]
alpha = [('Si','C',4.6417), ('Si',1.3642), ('C',2.555)]
re = [('Si','C',2.8429), ('Si',0.9862), ('C',2.522)]

# New coefficients from Granular_LJ_v4.ipynb
lj_const_r = [('Si', 'Si', 2.82865222e-03), ('Si', 'C', -9.41669755e-04), ('C', 'C', -1.02324090e-05)]
lj_const_a = [('Si', 'Si', 2.31721731e-01), ('Si', 'C', -3.37274534e-01), ('C', 'C', -3.30159241e-01)]
gauss_amplitude = [-1.55283232e+02, 4.28134381e+01, -9.87051902e+00, 4.39626367e+00]
add_const = -36.942677974149206

gauss_width = [1.0, 1.0, 1.0, 1.0]
gauss_center = [0.5, 1.5, 2.5, 3.5]

out_dict = {'sigma': sigma,
            'epsilon': epsilon,
            'de': de,
            'alpha': alpha,
            're': re,
            'lj_const_r': lj_const_r,
            'lj_const_a': lj_const_a,
            'gauss_amplitude': gauss_amplitude,
            'gauss_width': gauss_width,
            'gauss_center': gauss_center,
            'add_const': add_const}

with open('data.json', 'w') as f:
    json.dump(out_dict, f)