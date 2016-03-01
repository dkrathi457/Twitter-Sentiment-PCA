# Grab user information

# Load packages
library(twitteR)
library(dplyr)
library(rjson)
library(lubridate)

# Load data
today <- format(Sys.time(), '%Y-%m-%d')
searchstring <- 'microsoft'

# Load tweet file (from process_tweets.R)
files <- list.files('data',paste0('proc_',searchstring))
files
selectedfile <- paste0('data/',files[1])
statuses <- readRDS(file=selectedfile)

# Set up authorization
secrets <- fromJSON(file='twitter_secrets.json.nogit')
setup_twitter_oauth(secrets$api_key,
                    secrets$api_secret,
                    secrets$access_token,
                    secrets$access_token_secret)

# Grab user info
userlist <- sapply(unique(statuses$user), as.character)
allusers <- lookupUsers(userlist)

# Date scince founding of twitter (March 21, 2006)
twitter_date <- mdy_hm('03-21-2006 9:50 PM PST')

# Gather all the user info in a data frame
userinfo <- data.frame(user=sapply(allusers, function(x) x$screenName),
                       realname=sapply(allusers, function(x) x$name),
                       numstatuses=sapply(allusers, function(x) x$statusesCount),
                       followers=sapply(allusers, function(x) x$followersCount),
                       friends=sapply(allusers, function(x) x$friendsCount),
                       favorites=sapply(allusers, function(x) x$favoritesCount),
                       account_created=sapply(allusers, function(x) format(x$created, 
                                                              format='%F %T')),
                       verified=sapply(allusers, function(x) x$verified),
                       numlisted=sapply(allusers, function(x) x$listedCount)) %>%
    mutate(user=as.character(user)) %>%
    mutate(twitter_years=interval(twitter_date,account_created) / dyears(1)) %>%
    select(-account_created)

# Group tweet data by user
newstatuses <-
    statuses %>%
    group_by(user) %>%
    summarize(numTopicTweets=n(),
              positivity=mean(positivity),
              PC1=mean(PC1),
              PC2=mean(PC2),
              PC3=mean(PC3),
              PC4=mean(PC4),
              PC5=mean(PC5),
              client=rownames(sort(table(client), decreasing = T))[1], #most common client
              anger=mean(anger), anticipation=mean(anticipation), 
              disgust=mean(disgust), fear=mean(fear), joy=mean(joy),
              sadness=mean(sadness), surprise=mean(surprise), trust=mean(trust)) %>% 
    mutate(user=as.character(user))

# Join the data together
alldata <- inner_join(userinfo, newstatuses, by='user')
# looks like some users were lost

# Save to a file
savename <- paste0('data/users_',searchstring,'_',
                   nrow(alldata),'_',today,'.Rda')
saveRDS(alldata, file=savename)
