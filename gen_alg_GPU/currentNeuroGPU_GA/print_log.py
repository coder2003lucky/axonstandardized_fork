import pickle
fn = '../../Logs/10000/log.pkl'
#fn = './log.pkl'
f = open(fn, 'rb') 
log = pickle.load(f)
print(log)
