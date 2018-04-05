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
x <- matrix(0.0, 10, 45) #initialize to zeros

three <- docwords$Count[[9]]

x2<- matrix(0.0, 2,3)
x[2][2] <- 10

rowcount<-nrow(docwords)
print(rowcount)
for(i in 1:3) {
  row <- docwords[i,]
  doc <- row[[1]]
  print(paste0("doc ",doc))
  word <- row[[2]]
  print(paste0("word ",word))
  count <- row[[3]]
  print(paste0("count ",count))
  # do stuff with row
  x[doc][word]<-count
}


