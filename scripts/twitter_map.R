# Graph positions of tweets

library(ggplot2)
library(ggmap)
library(dplyr)

# Load the processed data
statuses <- readRDS(file='data/proc_tweets.Rda')

# Grab only those with location data
statuses <-
    statuses %>%
    filter(!is.na(latitude))

map <- get_map('United States', zoom=4)

ggmap(map) +
    geom_point(data=statuses, aes(x=longitude, y=latitude))
               