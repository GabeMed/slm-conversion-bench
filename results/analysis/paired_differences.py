import json,glob,random,math,sys
from collections import Counter,defaultdict
sys.path.insert(0,'<repo>')
R='<repo>/'
runs={'B0':'eval-agent-B0-test-20261111T060814.371294Z-20261111T080311.709677Z','B2c':'eval-agent-B2-cheap_alt-test-20261111T081959.861177Z-20261111T085323.154993Z','B2p':'eval-agent-B2-production_llm-test-20261111T144852.703285Z-20261111T154903.696240Z','B2p2':'eval-agent-B2-production_llm-test-20261111T144852.703285Z-20261111T202239.433284Z','B3':'eval-agent-B3-test-20261111T012223.392984Z-20261111T054533.117283Z','B4':'eval-agent-B4-test-20261111T032707.693347Z-20261111T054821.932671Z','B5':'eval-agent-B5-test-20261111T200809.693765Z-20261111T202046.064721Z'}
C={};meta={}
for a,r in runs.items():
    rows=[json.loads(l) for l in open(R+'runs/'+r+'/results.jsonl')]
    C[a]={str(x['question_id']):bool(x['correct']) for x in rows}
    meta={str(x['question_id']):x for x in rows}
    if a=='B0': print('keys',[k for k in rows[0].keys()])
    print(a,len(rows),sum(C[a].values()), 'pred_err',sum(1 for x in rows if x.get('pred_error')), 'gold_err',[x['question_id'] for x in rows if x.get('gold_error')], 'empty',sum(1 for x in rows if not (x.get('predicted_sql') or '').strip()))
print('B2p==B2p2',C['B2p']==C['B2p2'])
ids=sorted(C['B0'])
def binom_two(k,n):
    p=sum(math.comb(n,i) for i in range(0,min(k,n-k)+1))/2**n*2
    return min(1,p)
def binom_cdf(k,n,p):
    return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(0,k+1))
def cmp(a,b,sub=None,seed=20260932,nb=10000):
    q=sub or ids
    d=[int(C[a][i])-int(C[b][i]) for i in q]; n=len(d)
    ao,bo=d.count(1),d.count(-1)
    rng=random.Random(seed)
    means=sorted(sum(d[int(rng.random()*n)] for _ in range(n))/n for _ in range(nb))
    lo90,hi90=means[499],means[nb-500]; lo95,hi95=means[249],means[nb-250]
    diff=sum(d)/n
    # one-sided test H0: diff<=-0.05 via normal with paired var
    var=(ao+bo)/n-diff**2; se=math.sqrt(var/n)
    z=(diff+0.05)/se if se>0 else float('inf')
    from statistics import NormalDist
    p_ni=1-NormalDist().cdf(z)
    # Newcombe? use Wald 90
    print(f"{a}-{b} n={n} a_only={ao} b_only={bo} d={(ao+bo)/n:.3f} diff={100*diff:+.2f} boot90=[{100*lo90:+.2f},{100*hi90:+.2f}] boot95=[{100*lo95:+.2f},{100*hi95:+.2f}] wald90=[{100*(diff-1.645*se):+.2f},{100*(diff+1.645*se):+.2f}] McNemar exact p={binom_two(min(ao,bo),ao+bo):.2g} NI z-p={p_ni:.2g}")
    return ao,bo
for a,b in [('B5','B0'),('B4','B0'),('B4','B5'),('B4','B3'),('B3','B0'),('B2p','B0'),('B2c','B0'),('B2c','B2p'),('B5','B2p'),('B4','B2c')]:
    cmp(a,b)
print('--- without q875')
s=[i for i in ids if i!='875']
for a,b in [('B5','B0'),('B4','B0')]: cmp(a,b,s)
print('875:',{a:C[a].get('875') for a in C})
print('--- by difficulty')
dif=defaultdict(list)
for i in ids: dif[meta[i]['difficulty']].append(i)
for k,v in dif.items():
    print(k,len(v),{a:f"{sum(C[a][i] for i in v)}/{len(v)}={100*sum(C[a][i] for i in v)/len(v):.1f}" for a in ['B0','B2p','B2c','B3','B4','B5']})
    cmp('B5','B0',v); cmp('B4','B0',v)
print('--- by db')
db=defaultdict(list)
for i in ids: db[meta[i]['db_id']].append(i)
for k,v in sorted(db.items()):
    print(f"{k:28s} n={len(v):3d}",{a:sum(C[a][i] for i in v) for a in ['B0','B2p','B2c','B3','B4','B5']})
# power
from statistics import NormalDist
Z=NormalDist()
def power(d,n,m,true=0): return Z.cdf((m+true)/math.sqrt(d/n)-Z.inv_cdf(0.95))
for name,d in [('B5vB0 d',None)]: pass
for a,b in [('B5','B0'),('B4','B0'),('B3','B0'),('B2c','B0')]:
    d=sum(C[a][i]!=C[b][i] for i in ids)/498
    print(a,b,'obs d',round(d,3),'power@0',round(power(d,498,.05),3),'power@-2pp',round(power(d,498,.05,-.02),3),'power@-3pp',round(power(d,498,.05,-.03),3))
for d in [0.3,0.4,0.5,0.6]: print('d',d,'power',round(power(d,498,.05),3))
# all-arms agreement
print('all 6 correct',sum(all(C[a][i] for a in ['B0','B2p','B2c','B3','B4','B5']) for i in ids),'none',sum(not any(C[a][i] for a in ['B0','B2p','B2c','B3','B4','B5']) for i in ids))
print('B0 wrong & B2p wrong',sum((not C['B0'][i]) and (not C['B2p'][i]) for i in ids))
