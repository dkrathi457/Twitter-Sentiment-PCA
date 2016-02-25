# Load twitter data and save to file

library(twitteR)
library(dplyr)
library(rjson)

# Set up authorization
secrets <- fromJSON(file='twitter_secrets.json.nogit')
setup_twitter_oauth(secrets$api_key,
                    secrets$api_secret,
                    secrets$access_token,
                    secrets$access_token_secret)

# Search twitter
searchstring <- 'trump'
numtweets <- 10000
st <- searchTwitter(searchstring, n=numtweets, resultType = 'recent', lang = 'en')

statuses <- data.frame(text=sapply(st, function(x) x$getText()),
                       user=sapply(st, function(x) x$getScreenName()),
                       RT=sapply(st, function(x) x$isRetweet),
                       latitude=sapply(st, function(x) as.numeric(x$latitude[1])),
                       longitude=sapply(st, function(x) as.numeric(x$longitude[1])),
                       time=sapply(st, function(x) format(x$created, format='%F %T'))
                       )

# Remove retweets
statuses <-
    statuses %>%
    filter(!RT)

# Save tweets
saveRDS(statuses, file='tweet_data.Rda')