
path = '../data/ml-100k'
metapaths = ['umum', 'uaum', 'uoum', 'umtm']
#metapaths = ['ubub', 'ubib', 'ubab']
#metapaths = ['uaua', 'uua', 'uata']
#metapaths = ['umum', 'umcm', 'umbm', 'umvm']
print 'dir path : ', path
print 'metapaths : ', metapaths

sourcefile = path + '.um'
targetfile = path + '.label'

um_dict = {}
idx = -1
for metapath in metapaths:
    idx += 1
    print metapath
    with open(path + '.mp.' + metapath) as infile:
        for line in infile.readlines():
            arr = line.strip().split('\t')
            u, m = int(arr[0]), int(arr[1])
            if u not in um_dict:
                um_dict[u] = {}
            if m not in um_dict[u]:
                um_dict[u][m] = []
            um_dict[u][m].append(idx)

with open(targetfile, 'w') as outfile:
    for u in um_dict:
        for m in um_dict[u]:
            outfile.write(str(u) + ',' + str(m))
            for label in um_dict[u][m]:
                outfile.write(' ' + str(label))
            outfile.write('\n')

