library(kebabs)

argument = 2 # '' for only train and test file; 1 for thyme file1; 2 for thymefile2

aa = readAAStringSet(paste('../data/temporary_combined_file',argument,'.fa',sep=''))

specK7 = spectrumKernel(k=7,normalized=TRUE)

specFeat = getExRep(aa,kernel=specK7,sparse=FALSE)

write.csv(specFeat,file=gzfile(paste('../data/featfiles/trainfiles/spectrumKernel',argument,'.csv.gz',sep='')))

mismK3M1 = mismatchKernel(k=3,m=1,normalized=TRUE)

mismFeat = getExRep(aa,kernel=mismK3M1,sparse=FALSE)

write.csv(mismFeat,file=gzfile(paste('../data/featfiles/trainfiles/mismatchKernel',argument,'.csv.gz',sep='')))

gappyK1M2 = gappyPairKernel(k=3,m=2,normalized=TRUE)

gappyFeat=getExRep(aa,kernel=gappyK1M2,sparse=FALSE)

write.csv(gappyFeat,file=gzfile(paste('../data/featfiles/trainfiles/gappyKernel',argument,'.csv.gz',sep='')))
