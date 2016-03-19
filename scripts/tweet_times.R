# Twitter through time

library(dplyr)
library(ggplot2)
library(scales) # for date_time scales

# Load data
today <- format(Sys.time(), '%Y-%m-%d')
searchstring <- 'microsoft'

# Load tweet file (from process_tweets.R)
files <- list.files('data',paste0('proc_',searchstring))
files
selectedfile <- paste0('data/',files[1])
statuses <- readRDS(file=selectedfile)

statuses <-
    statuses %>%
    mutate(time = as.POSIXct(time, format='%F %T'))

# Removing text temporarily to check for duplicates
df <- 
    statuses %>%
    select(-text)
duplicateTweets <- duplicated(df)
statuses <- statuses[!duplicateTweets,]

ggplot(statuses, aes(time, PC2)) +
    geom_point(col='darkblue', alpha=0.2) + 
    scale_x_datetime(labels = date_format('%F %H:%M'),
                     breaks = date_breaks("30 min"),
                     limits = c(as.POSIXct("2016-02-29 20:00:00", format='%F %H'),
                                as.POSIXct("2016-03-01 03:00:00", format='%F %H'))) +
    labs(x='Date') + theme_bw() +
    theme(axis.text.x = element_text(angle=90))

