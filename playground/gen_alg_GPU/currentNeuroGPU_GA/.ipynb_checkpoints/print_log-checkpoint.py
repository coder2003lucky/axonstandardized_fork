import pickle
fn = './Logs/5000/log.pkl'

f = open(fn, 'rb') 
log = pickle.load(f)
print(log)