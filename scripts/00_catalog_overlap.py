import json,sys,collections
g=json.load(open(sys.argv[1],encoding='utf-8'))
fs=g['features'];print('features',len(fs))
def bb(f):
    import numpy as np;a=np.array(f['geometry']['coordinates'],dtype=object);c=np.array([q for q in __import__('itertools').chain.from_iterable([f['geometry']['coordinates'][0]] if f['geometry']['type']=='Polygon' else [r[0] for r in f['geometry']['coordinates']])]);xs=c[:,0];ys=c[:,1];return min(xs),min(ys),max(xs),max(ys)
box=(37.0428,37.4378,37.3307,37.5895)
hit=[]
for f in fs:
    x0,y0,x1,y1=bb(f);p=f['properties']
    if x1>=box[0] and x0<=box[2] and y1>=box[1] and y0<=box[3]:
        hit.append(p)
print('overlap',len(hit))
c=collections.Counter((p['datetime'][:10],p['platform'],p['catalog_id'],p['gsd'],p.get('view:off_nadir'),bool(p.get('pan_analytic')),bool(p.get('ms_analytic'))) for p in hit)
for k,v in sorted(c.items()):print(k,v)
