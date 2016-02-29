# Process tweets script

# Load packages
library(tm)
library(wordcloud)
library(dplyr)
library(caret)
library(knitr)
library(RColorBrewer)
library(stringr)
library(syuzhet) # for sentiment analysis

today <- format(Sys.time(), '%Y-%m-%d')
searchstring <- 'politics'

# Load tweet file (from load_twitter.R)
files <- list.files('data',paste0('tweets_',searchstring))
files
selectedfile <- paste0('data/',files[1])
statuses <- readRDS(file=selectedfile)


# Create corpus
textdata <- Corpus(VectorSource(statuses$text))

textdata <- 
    textdata %>%
    tm_map(removeWords, stopwords("english"), mc.cores=1) %>%
    tm_map(removePunctuation, mc.cores=1) %>%
    tm_map(content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),
           mc.cores=1) %>%
    tm_map(content_transformer(tolower), mc.cores=1) %>%
    tm_map(content_transformer(function(x) str_replace_all(x, "@\\w+", "")), 
           mc.cores=1) %>% # remove twitter handles
    tm_map(removeNumbers, mc.cores=1) %>%
    tm_map(removeWords, c('trump','realdonaldtrump'), mc.cores=1) %>%
    tm_map(stripWhitespace, mc.cores=1)

# Sentiment analysis
sentiments <- sapply(textdata, function(x) get_nrc_sentiment(as.character(x)))

sentiments <- as.data.frame(aperm(sentiments)) # transpose and save as dataframe
sentiments <- as.data.frame(lapply(sentiments, as.numeric)) # a bit more to organize
sentiments <-
    sentiments %>%
    mutate(positivity = positive - negative)

# Word cloud
pal2 <- brewer.pal(8,"Dark2")
savefig <- paste0('figures/words_',searchstring,'_',today,'.png')
png(savefig)
wordcloud(textdata, max.words = 100, colors= pal2, random.order=F, 
          rot.per=0.1, use.r.layout=F)
#ggsave(savefig)
dev.off()

# Organizing words
dtm <- DocumentTermMatrix(textdata)
dtm <- inspect(dtm)

words <- data.frame(term = colnames(dtm))
words$count <- colSums(dtm)

words <-
    words %>%
    arrange(desc(count))
head(words)

tweets <- as.data.frame(dtm)
ind <- data.frame('id'=seq.int(nrow(tweets)))
tweets <- cbind(ind, tweets)
rm(dtm, textdata) # clearing up memory

words_100 <- as.character(words[2:101,'term'])

tweets <- tweets[,c('id',words_100)]
head(tweets[,1:10])

# Run PCA
trans <- preProcess(tweets[,2:ncol(tweets)], method=c("pca"), thresh = 0.95)
pca <- predict(trans, tweets[,2:ncol(tweets)])

statuses <- cbind(statuses, pca[,1:5], sentiments)

# Plot PCA
pal2 <- brewer.pal(10,"RdBu")
p2 <- ggplot(statuses, aes(x=PC1, y=PC2)) + 
    geom_point(aes(fill=positivity), size=4, alpha=0.7, pch=21, stroke=1.3) + 
    scale_fill_gradientn(colours = pal2, limits=c(-5,5)) + theme_bw()
print(p2)
savefig <- paste0('figures/pca_',searchstring,'_',today,'.png')
ggsave(savefig, plot=p2)

# Additional PCA checks
loadings <- trans$rotation 
load_sqr <- loadings^2

load_sqr <- data.frame(load_sqr)
temp <- data.frame('term'=rownames(load_sqr))
load_sqr <- cbind(temp, load_sqr)
load_sqr %>%
    select(term, PC1) %>%
    arrange(desc(PC1)) %>%
    head(10)

load_sqr %>%
    select(term, PC2) %>%
    arrange(desc(PC2)) %>%
    head(10)

# Examine some tweets
#set.seed(42)
tweet_check <- function(text, pc, numbreaks=5){
    cuts <- cut(pc, numbreaks)
    temp <- data.frame(text=text, pc=pc, pc_val=cuts)
    temp <- temp %>%
        group_by(pc_val) %>%
        summarise(text=iconv(sample(text,1), to='UTF-8-MAC', sub='byte'))
    return(temp)
}

tweet_check(statuses$text, statuses$PC1, 10) %>% kable

tweet_check(statuses$text, statuses$PC2, 10) %>% kable

tweet_check(statuses$text, statuses$positivity, 10) %>% kable

# Save processed tweets
savename <- paste0('data/proc_',searchstring,'_',
                   nrow(statuses),'_',today,'.Rda')
saveRDS(statuses, file=savename)
