# Graph positions of tweets

library(ggplot2)
library(ggmap)
library(dplyr)

# Load data
today <- format(Sys.time(), '%Y-%m-%d')
searchstring <- 'microsoft'

# Load tweet file (from process_tweets.R)
files <- list.files('data',paste0('proc_',searchstring))
files
selectedfile <- paste0('data/',files[1])
statuses <- readRDS(file=selectedfile)

# Grab only those with location data
statuses <-
    statuses %>%
    filter(!is.na(latitude))

map <- get_map('United States', zoom=4)

ggmap(map) +
    geom_point(data=statuses, aes(x=longitude, y=latitude))
               