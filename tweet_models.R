# Script to fit models to the user data

# Load packages
library(dplyr)
library(caret)
library(ggplot2)
#library(rpart)
library(rattle)

# Load data
searchstring <- 'politics'
files <- list.files('data',paste0('users_',searchstring))
files
selectedfile <- paste0('data/',files[1])
alldata <- readRDS(file=selectedfile)

# Select predictor
choice <- 'PC2'
possiblepreds <- c('positivity','PC1','PC2','PC3','PC4','PC5')
colremove <- possiblepreds[sapply(possiblepreds, function(x) x!=choice)]
col_list <- colnames(alldata)
final_cols <- col_list[sapply(col_list, function(x) !(x %in% colremove))]

df <- alldata[final_cols] %>% 
    select(-user, -realname) %>%
    mutate(client=as.factor(client))

# Remove parameters with near zero variance
#nzv <- nearZeroVar(df, saveMetrics= TRUE)
#nzv[nzv$nzv,]
nzv <- nearZeroVar(df)
df_filter <- df[, -nzv]
df_filter <- na.omit(df_filter)

# Create training/test data
set.seed(3456)
trainIndex <- sample(nrow(df_filter), nrow(df_filter)*0.8)

df_train <- df_filter[ trainIndex,]
df_test  <- df_filter[-trainIndex,]

# Run models
df_test_all <- df_test
df_test_all[,'id'] <- seq(nrow(df_test_all))

## Regression Tree
if (T){
    rm(rtTune, rtGrid) # clear prior
    
    rtGrid = expand.grid(cp=seq(0.01, 0.2, by = 0.005)) # grid of cp values
    ctrl <- trainControl(method = "cv", number = 10, verboseIter = T)

    rtTune <- train(df_train[,choice] ~ ., data = df_train,   
                    method = "rpart", 
                    tuneGrid = rtGrid,
                    trControl = ctrl)
    
    #rtTune
    #plot(rtTune)
    #rtTune$bestTune
    
    fancyRpartPlot(rtTune$finalModel, palettes=c("Blues"), sub='')
    
    pr_rt <- predict(rtTune, newdata = df_test)
    #rt_CM <- confusionMatrix(pr_rt, df_test[,choice])
    #rt_CM
    rmseTree <- RMSE(pr_rt, df_test[,choice])

    df_test_all[,'diff_Tree'] = abs(df_test_all[,choice] - pr_rt)
}

ggplot(data=df_test_all, aes(x=id)) +
    geom_point(aes(y=diff_Tree), color='dark green', alpha=0.6) + 
    geom_hline(yintercept=rmseTree, color='dark green')

