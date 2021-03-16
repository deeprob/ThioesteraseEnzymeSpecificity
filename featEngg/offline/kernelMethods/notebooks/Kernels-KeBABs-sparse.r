library(kebabs)
library(Matrix)

argument = 2 # '' for only train and test file; 1 for thyme file1; 2 for thymefile2

aa = readAAStringSet(paste('../data/temporary_combined_file',argument,'.fa',sep=''))

specK7 = spectrumKernel(k=7,normalized=FALSE)
specFeat = getExRep(aa,kernel=specK7,sparse=TRUE)

matCSR_spec <- as(specFeat,"dgRMatrix")
write(colnames(matCSR_spec), file = paste("../data/featfiles/thymefiles",argument,"/spec_kern_colnames.txt",sep=""))
write(rownames(matCSR_spec), file = paste("../data/featfiles/thymefiles",argument,"/spec_kern_rownames.txt",sep=""))
writeMM(matCSR_spec, file = paste("../data/featfiles/thymefiles",argument,"/spec_kern_sparsematrix.txt",sep=""))

      
mismK3M1 = mismatchKernel(k=3,m=1,normalized=FALSE)
mismFeat = getExRep(aa,kernel=mismK3M1,sparse=TRUE)

matCSR_mism <- as(mismFeat,"dgRMatrix")
write(colnames(matCSR_mism), file = paste("../data/featfiles/thymefiles",argument,"/mism_kern_colnames.txt",sep=""))
write(rownames(matCSR_mism), file = paste("../data/featfiles/thymefiles",argument,"/mism_kern_rownames.txt",sep=""))
writeMM(matCSR_mism, file = paste("../data/featfiles/thymefiles",argument,"/mism_kern_sparsematrix.txt",sep=""))

gappyK1M2 = gappyPairKernel(k=3,m=2,normalized=FALSE)
gappyFeat=getExRep(aa,kernel=gappyK1M2,sparse=TRUE)

matCSR_gap <- as(gappyFeat,"dgRMatrix")
write(colnames(matCSR_gap), file = paste("../data/featfiles/thymefiles",argument,"/gap_kern_colnames.txt",sep=""))
write(rownames(matCSR_gap), file = paste("../data/featfiles/thymefiles",argument,"/gap_kern_rownames.txt",sep=""))
writeMM(matCSR_gap, file = paste("../data/featfiles/thymefiles",argument,"/gap_kern_sparsematrix.txt",sep=""))
