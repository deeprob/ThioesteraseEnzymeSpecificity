library(kebabs)

aa = readAAStringSet('../data/temporary_combined_file.fa')

specK7 = spectrumKernel(k=7,normalized=TRUE)

specFeat = getExRep(aa,kernel=specK7,sparse=FALSE)

write.csv(specFeat,file=gzfile('../data/featfiles/trainfiles/spectrumKernel.csv.gz'))

mismK3M1 = mismatchKernel(k=3,m=1,normalized=TRUE)

mismFeat = getExRep(aa,kernel=mismK3M1,sparse=FALSE)

write.csv(mismFeat,file=gzfile('../data/featfiles/trainfiles/mismatchKernel.csv.gz'))

gappyK1M2 = gappyPairKernel(k=3,m=2,normalized=TRUE)

gappyFeat=getExRep(aa,kernel=gappyK1M2,sparse=FALSE)

write.csv(gappyFeat,file=gzfile('../data/featfiles/trainfiles/gappyKernel.csv.gz'))
