library(kebabs)
library(Matrix)

train_aa = readAAStringSet('../data/seq/train_enz.fa')
test_aa = readAAStringSet('../data/seq/test_enz.fa')
aa = c(train_aa, test_aa)

specK7 = spectrumKernel(k=7,normalized=FALSE)
specFeat = getExRep(aa,kernel=specK7,sparse=TRUE)

matCSR_spec <- as(specFeat,"dgRMatrix")
write(colnames(matCSR_spec), file = "../data/features/kernel/spectrum/spec_kern_colnames.txt")
write(rownames(matCSR_spec), file = "../data/features/kernel/spectrum/spec_kern_rownames.txt")
writeMM(matCSR_spec, file = "../data/features/kernel/spectrum/spec_kern_sparsematrix.txt")

      
mismK3M1 = mismatchKernel(k=3,m=1,normalized=FALSE)
mismFeat = getExRep(aa,kernel=mismK3M1,sparse=TRUE)

matCSR_mism <- as(mismFeat,"dgRMatrix")
write(colnames(matCSR_mism), file = "../data/features/kernel/mismatch/mism_kern_colnames.txt")
write(rownames(matCSR_mism), file = "../data/features/kernel/mismatch/mism_kern_rownames.txt")
writeMM(matCSR_mism, file = "../data/features/kernel/mismatch/mism_kern_sparsematrix.txt")

gappyK1M2 = gappyPairKernel(k=3,m=2,normalized=FALSE)
gappyFeat=getExRep(aa,kernel=gappyK1M2,sparse=TRUE)

matCSR_gap <- as(gappyFeat,"dgRMatrix")
write(colnames(matCSR_gap), file = "../data/features/kernel/gappy/gap_kern_colnames.txt")
write(rownames(matCSR_gap), file = "../data/features/kernel/gappy/gap_kern_rownames.txt")
writeMM(matCSR_gap, file = "../data/features/kernel/gappy/gap_kern_sparsematrix.txt")
