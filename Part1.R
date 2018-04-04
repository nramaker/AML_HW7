#load document
params <- read.csv('./NIPS/docword.nips.txt', header=FALSE, sep=" ", nrows=3)
docwords <- read.csv('./NIPS/docword.nips.txt', header=FALSE, sep=" ", skip=3)
vocab <-read.csv('./NIPS/vocab.nips.txt', header=FALSE, sep=' ')
colnames(docwords) <- c("DocID", "WordID", "Count")

d <-params[1,] #number of documents
w <-params[2,] #number of words in the vocabulary
n <-params[3,] #total number of words
k <-30  #number of clusters

#build my vectors of x's.  This should be d vectors of w length
x1 <- rep(0.0, w)
x <- matrix(0.0, d, w) #initialize to zeros

three <- docwords$Count[[9]]


rowcount<-nrow(docwords)
print(rowcount)
for(i in 1:3) {
  row <- docwords[i,]
  doc <- row[[1]]
  word <- row[[2]]
  count <- row[[3]]
  #print(count)
  # do stuff with row
  x[[doc]][[word]]<-count
}


