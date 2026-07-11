##############################################################
#  LEH cleaning code
#  Don Moore
##############################################################

library(groundhog)
library(tidyr)
library(dplyr)
groundhog.library('ggplot2','2026-01-13')
groundhog.library(reshape2, '2026-01-13')
stderr <- function(x) sqrt(var(x,na.rm=TRUE)/length(na.omit(x))) #define standard error function
rstudioapi::writeRStudioPreference("data_viewer_max_columns", 1000L) #change the number of columns shown from 50

rm(list=ls())                      # clear environment
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014")                        # clear console

# Importing Data from Qualtrics CSV file
### Note: Must download the data from Qualtrics with the "Use numeric values" option selected, and checking the box to "Export viewing order data for randomized surveys"
#then manually delete rows 2-3 leaving only one header row
allData = read.csv("LEH_January+13,+2026_14.26.csv", na.strings = c("", "NA"))[-c(1,2),]
#allData <- read.csv("211013HBAR.csv",header=TRUE, na.strings = c("", "NA")) # Import Qualtrics data
# then export dataframe and read in data again to auto-class variables
# export df
write.csv(allData,"temp.csv")
# read in df
allData <- read.csv("temp.csv")

##clean the data
#drop incomplete rows and people who fail attention checks
valid <- subset(allData,Finished == TRUE) #keep only those who completed the survey
valid <- subset(valid,Duration..in.seconds. > 120) #drop those who took less than 2 min
valid <- valid[grepl("plant", data$FILTER3), ] #drop those who got the attention check wrong

##Keep only data columns
dfWide <- valid[, c(9, 15:190)]

####Reorganize the wide data to long
# Read the data
temp <- dfWide
questions <- read.csv("260113LifeEvalQuestions.csv")

# Reshape temp from wide to long format
temp$X <- NULL  # Remove the index column if it exists
# First, convert to long format
temp_long <- temp %>%
  pivot_longer(
    cols = -ResponseId,
    names_to = "variable",
    values_to = "value"
  ) %>%
  filter(!is.na(value))  # Remove NA values (cases not answered)

# Extract case number and question type from variable names
temp_long <- temp_long %>%
  mutate(
    # Extract the case number (X1, X2, etc.) and question (Q195 or Q196_1)
    case_num = as.numeric(gsub("X([0-9]+)_Q.*", "\\1", variable)),
    question_type = ifelse(grepl("Q195", variable), "age", "certainty")
  )

# Pivot wider to get age and certainty as separate columns
temp_wide <- temp_long %>%
  select(ResponseId, case_num, question_type, value) %>%
  pivot_wider(
    names_from = question_type,
    values_from = value
  )

# Merge with the questions file to get the attributes
# The case_num corresponds to Row in the questions file
final_data <- temp_wide %>%
  left_join(questions, by = c("case_num" = "Row")) %>%
  select(ResponseId, case_num, age, certainty, MinAge, Radius, SexMale, LifeEvalQ) %>%
  arrange(ResponseId, case_num)

# View the result
head(final_data, 20)

# Save if desired
write.csv(final_data, "reorganized_data.csv", row.names = FALSE)