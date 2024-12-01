# Clear the environment
rm(list = ls())

# Load necessary libraries
#library(Matrix)
#library(glmnet)
library(ggplot2)
library(caret)
library(corrplot)
library(GGally)
library(ggcorrplot)
#library(Amelia)
library(psych)
library(dplyr)
#library(doParallel)
library(car)          # For VIF calculation
library(nnet)         # For multinomial logistic regression
#library(smotefamily)  # For SMOTE
library(vcd)          # For Cramér's V calculation
library(reshape2)     # For reshaping data in the ANOVA p-value heatmap
#library(MASS)
library(DMwR2)         # For SMOTE
library(randomForest) # For Random Forest
library(nnet)
library(caret)
#library(gridExtra)

set.seed(123)

# Set working directory
setwd("~/Desktop/GTX/MGT6203/Health_MGT6203_Project")

# Load the dataset
file_path <- "./health_data.csv"
data <- read.csv(file_path)

# Remove unnecessary columns, including GeneticRisk
data <- data %>% dplyr::select(-PatientID, -GeneticRisk)

# Split BloodPressure into Systolic and Diastolic Pressure
data <- data %>%
  mutate(SystolicPressure = as.numeric(sub("/.*", "", BloodPressure)),
         DiastolicPressure = as.numeric(sub(".*/", "", BloodPressure))) %>%
  dplyr::select(-BloodPressure)

# Convert AnnualCheckups to a factor with reference level 4
data$AnnualCheckups <- relevel(as.factor(data$AnnualCheckups), ref = "4")

# Convert categorical variables to factors
categorical_vars <- c("Gender", "SmokingStatus", "AlcoholConsumption", "ExerciseFrequency", 
                      "Diabetes", "HeartDisease", "PhysicalActivityLevel", 
                      "DietQuality", "MedicationAdherence", "Outcome")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Count unique items in each category for categorical columns
category_counts <- lapply(categorical_vars, function(col) {
  as.data.frame(table(data[[col]]))
})

# Name the list with column names
names(category_counts) <- categorical_vars

# Print category counts for each categorical variable
category_counts

# Save category counts to individual CSV files
for (col_name in names(category_counts)) {
  write.csv(category_counts[[col_name]], paste0(col_name, "_counts.csv"), row.names = FALSE)
}

# Create and display separate bar plots for each categorical variable
for (col in categorical_vars) {
  plot <- ggplot(data, aes_string(x = col)) +
    geom_bar(fill = "lightblue", color = "black") +
    labs(title = paste("Distribution of", col), x = col, y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10))
  
  # Print the plot for the current column
  print(plot)
}

# Check and visualize missing values
missing_data <- colSums(is.na(data))
print(missing_data)

# --------------------------------------------
# Chi-square Test
# --------------------------------------------

# Loop through all categorical variables except Outcome to test their association with Outcome
for (col in categorical_vars[-which(categorical_vars == "Outcome")]) {
  # Create contingency table for each variable with Outcome
  table_data <- table(data[[col]], data$Outcome)
  
  # Perform Chi-square test with Monte Carlo simulation to handle small expected counts
  chi_test <- chisq.test(table_data, simulate.p.value = TRUE, B = 10000) # B is the number of simulations
  
  # Print variable name and test result
  cat("Variable:", col, "\n")
  print(chi_test)
  cat("\n")
}

# --------------------------------------------
# A Detailed Look into data
# --------------------------------------------

# Inspect the structure of the dataset
str(data)
head(data)

# 'data.frame':	300000 obs. of  17 variables:
#   $ Age                  : int  82 62 26 64 34 34 46 71 38 26 ...
# $ Gender               : Factor w/ 3 levels "Female","Male",..: 2 1 1 2 1 2 2 1 1 1 ...
# $ BMI                  : num  27.7 26.1 37.6 33.1 30.7 ...
# $ SmokingStatus        : Factor w/ 3 levels "Current","Former",..: 3 2 3 1 3 3 2 3 2 3 ...
# $ AlcoholConsumption   : Factor w/ 3 levels "Never","Occasionally",..: 1 1 2 1 3 1 1 1 2 2 ...
# $ ExerciseFrequency    : Factor w/ 4 levels "Never","Often",..: 1 4 1 4 4 4 2 2 3 3 ...
# $ CholesterolLevel     : num  171 184 257 163 247 ...
# $ Diabetes             : Factor w/ 2 levels "No","Yes": 1 2 1 1 1 2 1 2 1 2 ...
# $ HeartDisease         : Factor w/ 2 levels "No","Yes": 2 1 1 1 1 1 1 1 1 1 ...
# $ PhysicalActivityLevel: Factor w/ 3 levels "High","Low","Medium": 2 2 3 3 3 3 1 1 2 1 ...
# $ DietQuality          : Factor w/ 3 levels "Average","Good",..: 1 2 1 1 1 3 1 3 1 2 ...
# $ MedicationAdherence  : Factor w/ 3 levels "High","Low","Medium": 3 3 1 3 1 3 3 2 3 3 ...
# $ AnnualCheckups       : Factor w/ 5 levels "4","0","1","2",..: 4 3 3 5 1 5 3 3 3 1 ...
# $ HealthcareCost       : num  20975 12693 13104 15920 18530 ...
# $ Outcome              : Factor w/ 3 levels "At Risk","Critical",..: 3 3 2 3 2 3 3 3 3 3 ...
# $ SystolicPressure     : num  124 155 100 159 131 144 168 141 119 159 ...
# $ DiastolicPressure    : num  80 75 96 82 80 93 113 98 118 112 ...

# Summary statistics to understand data distribution
summary(data)

# Age           Gender            BMI        SmokingStatus       AlcoholConsumption ExerciseFrequency
# Min.   :18.00   Female:147675   Min.   :15.00   Current: 59670   Never       :150089   Never    :60048  
# 1st Qu.:35.00   Male  :146234   1st Qu.:26.63   Former : 60023   Occasionally: 90013   Often    :60236  
# Median :53.00   Other :  6091   Median :30.13   Never  :180307   Regularly   : 59898   Rarely   :89841  
# Mean   :53.49                   Mean   :30.07                                          Sometimes:89875  
# 3rd Qu.:72.00                   3rd Qu.:33.62                                                           
# Max.   :89.00                   Max.   :40.00                                                           
# CholesterolLevel Diabetes     HeartDisease PhysicalActivityLevel  DietQuality     MedicationAdherence
# Min.   :100.0    No :254961   No :270214   High  : 59683         Average:150339   High  : 89680      
# 1st Qu.:192.5    Yes: 45039   Yes: 29786   Low   :119921         Good   : 59942   Low   : 59913      
# Median :214.5                              Medium:120396         Poor   : 89719   Medium:150407      
# Mean   :214.6                                                                                        
# 3rd Qu.:236.6                                                                                        
# Max.   :300.0                                                                                        
# AnnualCheckups HealthcareCost      Outcome       SystolicPressure DiastolicPressure
# 4:60187        Min.   :  500   At Risk : 27269   Min.   : 90.0    Min.   : 60.00   
# 0:59964        1st Qu.: 7740   Critical: 55284   1st Qu.:112.0    1st Qu.: 73.00   
# 1:60229        Median :11969   Healthy :217447   Median :134.0    Median : 86.00   
# 2:59861        Mean   :12434                     Mean   :134.5    Mean   : 86.91   
# 3:59759        3rd Qu.:16664                     3rd Qu.:157.0    3rd Qu.:100.00   
# Max.   :44761                     Max.   :179.0    Max.   :119.00 

# Additional descriptive statistics
describe(data)

# --------------------------------------------
# Outlier Detection Using Z-Score Method
# --------------------------------------------

# Calculate Z-scores for each numerical column
numeric_columns <- sapply(data, is.numeric)
z_scores <- as.data.frame(scale(data[, numeric_columns]))

# Identify outliers (where absolute Z-score > 3) and count them per column
outlier_counts <- sapply(z_scores, function(x) sum(abs(x) > 3))
print("Outlier Counts per Column:")
print(outlier_counts)

# Identify rows with any Z-score > 3 and get the original values
outliers <- data[apply(z_scores, 1, function(row) any(abs(row) > 3)), ]

# Create a column indicating which specific columns have outliers for each row
outliers$Outlier_Columns <- apply(z_scores[apply(z_scores, 1, function(row) any(abs(row) > 3)), ], 1, function(row) {
  paste(names(row)[abs(row) > 3], collapse = ", ")
})

# Display outlier counts and save outliers with identifying columns to a CSV file
write.csv(outliers, file = "z_score_outliers_with_columns.csv", row.names = FALSE)

# Age               BMI  CholesterolLevel    HealthcareCost  SystolicPressure DiastolicPressure 
# 0                 0               416               761                 0                 0 

# --------------------------------------------
# Yeo-Johnson Transformation for Outlier Columns
# --------------------------------------------

# Box-Cox, Square Root, and Log Transformation were tested and did not improve the outlier situation 

# Winsorization can be further used to cap the high end for 
# HealthcareCost and the low end for CholesterolLevel if needed.

# Find the optimal lambda for CholesterolLevel
cholesterol_lambda <- powerTransform(data$CholesterolLevel + 1, family = "yjPower")$lambda  # Add 1 to handle zeros
# Apply the Yeo-Johnson transformation using the optimal lambda
data$CholesterolLevel_YeoJohnson <- yjPower(data$CholesterolLevel + 1, lambda = cholesterol_lambda)

# Find the optimal lambda for HealthcareCost
healthcare_lambda <- powerTransform(data$HealthcareCost + 1, family = "yjPower")$lambda
# Apply the Yeo-Johnson transformation using the optimal lambda
data$HealthcareCost_YeoJohnson <- yjPower(data$HealthcareCost + 1, lambda = healthcare_lambda)

# Re-calculate Z-scores for each numerical column after Winsorization
numeric_columns <- sapply(data, is.numeric)
z_scores <- as.data.frame(scale(data[, numeric_columns]))

# Identify remaining outliers (where absolute Z-score > 3) and count them per column
remaining_outlier_counts <- sapply(z_scores, function(x) sum(abs(x) > 3))
print(remaining_outlier_counts)

# Filter rows with any remaining Z-score > 3 and get the original values for review
remaining_outliers <- data[apply(z_scores, 1, function(row) any(abs(row) > 3)), ]

# Display remaining outlier counts and save results for detailed examination
write.csv(remaining_outliers, file = "remaining_z_score_outliers.csv", row.names = FALSE)

# print(remaining_outlier_counts)
# Age                         BMI            CholesterolLevel              HealthcareCost 
# 0                           0                         416                         761 
# SystolicPressure           DiastolicPressure CholesterolLevel_YeoJohnson   HealthcareCost_YeoJohnson 
# 0                           0                         388                          87 

# --------------------------------------------
# Create data_scaled After Dealing with Outliers 
# --------------------------------------------

num_cols <- sapply(data, is.numeric)
data_scaled <- data
data_scaled[, num_cols] <- scale(data[, num_cols])

# Check and visualize missing values
missing_data_scaled <- colSums(is.na(data_scaled))
print(missing_data_scaled)

# --------------------------------------------
# A Detailed Look into data_scaled
# --------------------------------------------

# Remove unnecessary columns after box-cox transform
data_scaled <- data_scaled %>% dplyr::select(-CholesterolLevel, -HealthcareCost)

# Inspect the structure of the dataset
str(data_scaled)
head(data_scaled)

# 'data.frame':	300000 obs. of  17 variables:
#   $ Age                        : num  1.371 0.409 -1.322 0.506 -0.937 ...
# $ Gender                     : Factor w/ 3 levels "Female","Male",..: 2 1 1 2 1 2 2 1 1 1 ...
# $ BMI                        : num  -0.464 -0.782 1.497 0.605 0.119 ...
# $ SmokingStatus              : Factor w/ 3 levels "Current","Former",..: 3 2 3 1 3 3 2 3 2 3 ...
# $ AlcoholConsumption         : Factor w/ 3 levels "Never","Occasionally",..: 1 1 2 1 3 1 1 1 2 2 ...
# $ ExerciseFrequency          : Factor w/ 4 levels "Never","Often",..: 1 4 1 4 4 4 2 2 3 3 ...
# $ Diabetes                   : Factor w/ 2 levels "No","Yes": 1 2 1 1 1 2 1 2 1 2 ...
# $ HeartDisease               : Factor w/ 2 levels "No","Yes": 2 1 1 1 1 1 1 1 1 1 ...
# $ PhysicalActivityLevel      : Factor w/ 3 levels "High","Low","Medium": 2 2 3 3 3 3 1 1 2 1 ...
# $ DietQuality                : Factor w/ 3 levels "Average","Good",..: 1 2 1 1 1 3 1 3 1 2 ...
# $ MedicationAdherence        : Factor w/ 3 levels "High","Low","Medium": 3 3 1 3 1 3 3 2 3 3 ...
# $ AnnualCheckups             : Factor w/ 5 levels "4","0","1","2",..: 4 3 3 5 1 5 3 3 3 1 ...
# $ Outcome                    : Factor w/ 3 levels "At Risk","Critical",..: 3 3 2 3 2 3 3 3 3 3 ...
# $ SystolicPressure           : num  -0.404 0.79 -1.328 0.944 -0.134 ...
# $ DiastolicPressure          : num  -0.415 -0.715 0.546 -0.295 -0.415 ...
# $ CholesterolLevel_YeoJohnson: num  -1.357 -0.952 1.296 -1.571 1.006 ...
# $ HealthcareCost_YeoJohnson  : num  1.252 0.126 0.187 0.586 0.937 ...

# Summary statistics to understand data distribution
summary(data_scaled)

# Age              Gender            BMI          SmokingStatus       AlcoholConsumption ExerciseFrequency
# Min.   :-1.70703   Female:147675   Min.   :-2.9881   Current: 59670   Never       :150089   Never    :60048  
# 1st Qu.:-0.88935   Male  :146234   1st Qu.:-0.6819   Former : 60023   Occasionally: 90013   Often    :60236  
# Median :-0.02356   Other :  6091   Median : 0.0116   Never  :180307   Regularly   : 59898   Rarely   :89841  
# Mean   : 0.00000                   Mean   : 0.0000                                          Sometimes:89875  
# 3rd Qu.: 0.89032                   3rd Qu.: 0.7050                                                           
# Max.   : 1.70801                   Max.   : 1.9694                                                           
# Diabetes     HeartDisease PhysicalActivityLevel  DietQuality     MedicationAdherence AnnualCheckups
# No :254961   No :270214   High  : 59683         Average:150339   High  : 89680       4:60187       
# Yes: 45039   Yes: 29786   Low   :119921         Good   : 59942   Low   : 59913       0:59964       
# Medium:120396         Poor   : 89719   Medium:150407       1:60229       
# 2:59861       
# 3:59759       
# 
# Outcome       SystolicPressure   DiastolicPressure  CholesterolLevel_YeoJohnson HealthcareCost_YeoJohnson
# At Risk : 27269   Min.   :-1.71332   Min.   :-1.61611   Min.   :-3.498203           Min.   :-2.27277         
# Critical: 55284   1st Qu.:-0.86608   1st Qu.:-0.83550   1st Qu.:-0.681048           1st Qu.:-0.65632         
# Healthy :217447   Median :-0.01884   Median :-0.05488   Median :-0.004397           Median : 0.01863         
# Mean   : 0.00000   Mean   : 0.00000   Mean   : 0.000000           Mean   : 0.00000         
# 3rd Qu.: 0.86691   3rd Qu.: 0.78578   3rd Qu.: 0.676041           3rd Qu.: 0.68747         
# Max.   : 1.71416   Max.   : 1.92668   Max.   : 2.644762           Max.   : 3.90836  

# Additional descriptive statistics
describe(data_scaled)

# --------------------------------------------
# Outlier Visualization Exploratory Data Analysis (EDA)
# --------------------------------------------

# Visualize Outliers and Numerical Distributions
create_boxplot <- function(column_name) {
  column_data <- data[[column_name]]
  p15 <- quantile(column_data, 0.15, na.rm = TRUE)
  p85 <- quantile(column_data, 0.85, na.rm = TRUE)
  
  ggplot(data, aes(x = "", y = !!sym(column_name))) +
    geom_boxplot(outlier.colour = "red", outlier.shape = 16) +
    coord_cartesian(ylim = c(p15, p85)) +
    labs(title = paste("Box Plot (15th - 85th percentile) of", column_name), y = column_name, x = "") +
    theme_minimal()
}

create_distribution_plot <- function(column_name) {
  ggplot(data, aes(x = !!sym(column_name))) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, color = "black", fill = "blue", alpha = 0.7) +
    geom_density(color = "red", linewidth = 1) +
    labs(title = paste("Distribution of", column_name), x = column_name, y = "Density") +
    theme_minimal()
}

numeric_column_names <- names(data_scaled)[sapply(data_scaled, is.numeric)]
for (col in numeric_column_names) {
  print(create_boxplot(col))
  print(create_distribution_plot(col))
}
# --------------------------------------------
# Continue Exploratory Data Analysis (EDA)
# --------------------------------------------

# --------------------------------------------
# Multicollinearity Check using VIF
# --------------------------------------------

predictors <- data_scaled %>% dplyr::select(-Outcome)
vif_model <- lm(as.numeric(runif(nrow(predictors))) ~ ., data = predictors)
vif_values <- vif(vif_model)
print(vif_values)

# --------------------------------------------
# Correlation Analysis
# --------------------------------------------

# --------------------------------------------
# Numerical Correlation Matrix with ggcorrplot
# --------------------------------------------
num_cols <- sapply(data_scaled, is.numeric)
numeric_data <- data_scaled[, num_cols]
num_corr_matrix <- cor(numeric_data, use = "complete.obs")

# Plot using ggcorrplot
ggcorrplot(num_corr_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE, 
           title = "Numerical Correlation Matrix", 
           lab_size = 3, 
           colors = c("red", "white", "blue"),
           outline.col = "black") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))

# --------------------------------------------
# Categorical Correlation Matrix (Cramér's V) with ggplot2
# --------------------------------------------
# Calculate Cramér's V
calculate_cramers_v <- function(x, y) {
  tbl <- table(x, y)
  cramers_v <- assocstats(tbl)$cramer
  return(cramers_v)
}

cat_corr_matrix <- matrix(NA, nrow = length(categorical_vars), ncol = length(categorical_vars),
                          dimnames = list(categorical_vars, categorical_vars))

for (i in 1:length(categorical_vars)) {
  for (j in i:length(categorical_vars)) {
    if (i == j) {
      cat_corr_matrix[i, j] <- 1
    } else {
      cat_corr_matrix[i, j] <- calculate_cramers_v(data[[categorical_vars[i]]], data[[categorical_vars[j]]])
      cat_corr_matrix[j, i] <- cat_corr_matrix[i, j]
    }
  }
}

# Convert categorical correlation matrix to long format for ggplot2
cat_corr_df <- as.data.frame(as.table(cat_corr_matrix))
colnames(cat_corr_df) <- c("Var1", "Var2", "CramersV")

# Plot categorical correlation matrix
ggplot(cat_corr_df, aes(Var1, Var2, fill = CramersV)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "brown") +
  labs(title = "Categorical Correlation Matrix (Cramér's V)", x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14))

# --------------------------------------------
# ANOVA p-values Heatmap with ggplot2
# --------------------------------------------
anova_pvalues <- sapply(categorical_vars, function(cat_var) {
  sapply(names(data_scaled[, num_cols]), function(num_var) {
    model <- aov(data_scaled[[num_var]] ~ data[[cat_var]])
    summary(model)[[1]][["Pr(>F)"]][1]
  })
})

anova_pvalues_df <- as.data.frame(anova_pvalues)
anova_pvalues_df$Categorical <- rownames(anova_pvalues_df)

# Reshape for ggplot2 heatmap
anova_pvalues_melted <- melt(anova_pvalues_df, id.vars = "Categorical", 
                             variable.name = "Numerical", value.name = "p_value")

# Plot ANOVA heatmap
ggplot(anova_pvalues_melted, aes(x = Numerical, y = Categorical, fill = p_value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "red", high = "blue", na.value = "gray", limits = c(0, 1)) +
  labs(title = "ANOVA p-values for Categorical vs Numerical Variables",
       x = "Numerical Variable", y = "Categorical Variable") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(size = 14, hjust = 0.5))

# --------------------------------------------
# Multinomial Logistic Regression
# --------------------------------------------

# Fit a multinomial logistic regression model
multinom_model <- multinom(Outcome ~ ., data = data_scaled)
summary(multinom_model)

# Result
# CholesterolLevel_YeoJohnson: Positive for Critical, negative for Healthy, indicating it’s an influential variable.
# HealthcareCost_YeoJohnson: Large positive coefficient for Critical and large negative for Healthy.
# Other predictors with moderate effects include ExerciseFrequency, Diabetes, HeartDisease, and Age.

# Call:
#   multinom(formula = Outcome ~ ., data = data_scaled)
# 
# Coefficients:
#   (Intercept)          Age   GenderMale GenderOther         BMI SmokingStatusFormer SmokingStatusNever
# Critical  -0.4046756 -0.013458527 -0.003740396 -0.02242749  0.04576618         -0.06229575        -0.05520769
# Healthy    2.1777519 -0.004203188  0.007661245 -0.07478334 -0.05925955         -0.02860764        -0.02871230
# AlcoholConsumptionOccasionally AlcoholConsumptionRegularly ExerciseFrequencyOften ExerciseFrequencyRarely
# Critical                   -0.025652285                 -0.02244653            0.031092973             -0.01223337
# Healthy                    -0.009185105                 -0.01526903            0.003944318             -0.02131258
# ExerciseFrequencySometimes DiabetesYes HeartDiseaseYes PhysicalActivityLevelLow PhysicalActivityLevelMedium
# Critical                0.028318962  0.01102517     -0.01299593             -0.008936187                -0.005418870
# Healthy                 0.008226126  0.02089287     -0.01724235              0.003255437                 0.004342537
# DietQualityGood DietQualityPoor MedicationAdherenceLow MedicationAdherenceMedium AnnualCheckups0 AnnualCheckups1
# Critical     0.003008011     -0.01214797           -0.010906902             -0.0006267888    0.0164009009    -0.003276743
# Healthy     -0.009007843      0.01212582           -0.003865583              0.0004841969    0.0007175832    -0.006095722
# AnnualCheckups2 AnnualCheckups3 SystolicPressure DiastolicPressure CholesterolLevel_YeoJohnson
# Critical      0.02851027     0.005122196     0.0052525025        0.00717348                   0.3248581
# Healthy       0.03211623     0.002734320     0.0009631218        0.01429737                  -0.3232643
# HealthcareCost_YeoJohnson
# Critical                  1.350944
# Healthy                  -1.194635
# 
# Std. Errors:
#   (Intercept)         Age GenderMale GenderOther         BMI SmokingStatusFormer SmokingStatusNever
# Critical  0.03848283 0.007869259 0.01590210  0.05529313 0.007895623          0.02497230         0.02043538
# Healthy   0.03217800 0.006716557 0.01357919  0.04731404 0.006733880          0.02137034         0.01751835
# AlcoholConsumptionOccasionally AlcoholConsumptionRegularly ExerciseFrequencyOften ExerciseFrequencyRarely
# Critical                     0.01815888                  0.02081955             0.02486517              0.02267737
# Healthy                      0.01550979                  0.01775148             0.02125735              0.01933699
# ExerciseFrequencySometimes DiabetesYes HeartDiseaseYes PhysicalActivityLevelLow PhysicalActivityLevelMedium
# Critical                 0.02276491  0.02207962      0.02631571               0.02154781                  0.02154054
# Healthy                  0.01943200  0.01889832      0.02241536               0.01842431                  0.01842315
# DietQualityGood DietQualityPoor MedicationAdherenceLow MedicationAdherenceMedium AnnualCheckups0 AnnualCheckups1
# Critical      0.02077550      0.01820215             0.02272515                0.01818934      0.02480735      0.02477066
# Healthy       0.01775831      0.01553075             0.01940390                0.01552884      0.02117982      0.02112539
# AnnualCheckups2 AnnualCheckups3 SystolicPressure DiastolicPressure CholesterolLevel_YeoJohnson
# Critical      0.02493461      0.02483657      0.008043682       0.008041932                 0.008169064
# Healthy       0.02130708      0.02119364      0.006863835       0.006864068                 0.006946188
# HealthcareCost_YeoJohnson
# Critical               0.011683704
# Healthy                0.009211196
# 
# Residual Deviance: 320511.9 
# AIC: 320623.9 


# --------------------------------------------
# Multinomial Logistic Regression with Select Variables
# Divide the data into train, test, validate
# --------------------------------------------

# Filter out low-impact variables
# Selected high-impact variables based on initial model results
selected_vars <- c("Outcome", "Age", "CholesterolLevel_YeoJohnson", "HealthcareCost_YeoJohnson",
                   "ExerciseFrequency", "Diabetes", "HeartDisease")

# Create a new dataset with only selected variables
data_refined <- data_scaled[, selected_vars]

# Split the refined dataset into Training, Validation, and Test sets
train_index <- createDataPartition(data_refined$Outcome, p = 0.7, list = FALSE)
train_data <- data_refined[train_index, ]
test_val_data <- data_refined[-train_index, ]

# Further split into validation and test sets
val_index <- createDataPartition(test_val_data$Outcome, p = 0.5, list = FALSE)
val_data <- test_val_data[val_index, ]
test_data <- test_val_data[-val_index, ]

# Refit Multinomial Logistic Regression Model on Reduced Variables
# Set up 10-fold cross-validation for model evaluation
control <- trainControl(method = "cv", number = 10)

# Fit the multinomial logistic regression model with selected variables
multinom_refined_model <- train(
  Outcome ~ ., data = train_data, method = "multinom",
  trControl = control
)

# Evaluate Model Performance on Validation Set
val_predictions <- predict(multinom_refined_model, newdata = val_data)
val_accuracy <- mean(val_predictions == val_data$Outcome)
cat("Validation Accuracy of Refined Model:", val_accuracy, "\n")

# Evaluate Final Model on Test Set
test_predictions <- predict(multinom_refined_model, newdata = test_data)
conf_matrix <- confusionMatrix(test_predictions, test_data$Outcome)
print(conf_matrix)


# Validation Accuracy of Refined Model: 0.8034222 
# 
# 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction At Risk Critical Healthy
# At Risk        0        0       0
# Critical    1060     5137    1643
# Healthy     3030     3155   30974
# 
# Overall Statistics
# 
# Accuracy : 0.8025          
# 95% CI : (0.7988, 0.8062)
# No Information Rate : 0.7248          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4652          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: At Risk Class: Critical Class: Healthy
# Sensitivity                 0.00000          0.6195         0.9496
# Specificity                 1.00000          0.9264         0.5005
# Pos Pred Value                  NaN          0.6552         0.8336
# Neg Pred Value              0.90911          0.9151         0.7904
# Prevalence                  0.09089          0.1843         0.7248
# Detection Rate              0.00000          0.1142         0.6883
# Detection Prevalence        0.00000          0.1742         0.8258
# Balanced Accuracy           0.50000          0.7729         0.7251


# --------------------------------------------
# Deal with imbalance in prediction 
# --------------------------------------------

# Downsample the majority class (Healthy) to reduce the size of the dataset
# First, check class distribution
table(train_data$Outcome)
# At Risk Critical  Healthy 
# 19089    19089    19089 

# Downsample the majority class 'Healthy'
train_data_downsampled <- train_data %>%
  group_by(Outcome) %>%
  sample_n(min(table(train_data$Outcome)), replace = FALSE) %>%
  ungroup()

# Check the new class distribution after downsampling
cat("Class distribution after downsampling:\n")
print(table(train_data_downsampled$Outcome))

# Define cross-validation with SMOTE applied directly in caret
control <- trainControl(method = "cv", number = 10, sampling = "smote")  # SMOTE will be applied during CV

# Train the Random Forest model with cross-validation
rf_model <- train(
  Outcome ~ ., data = train_data_downsampled, method = "rf",
  trControl = control, tuneLength = 5  # tuneLength controls the search grid for mtry
)
print(rf_model)  # Display model summary

# Random Forest 
# 
# 57267 samples
# 6 predictor
# 3 classes: 'At Risk', 'Critical', 'Healthy' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 51540, 51540, 51541, 51540, 51541, 51540, ... 
# Addtional sampling using SMOTE
# 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2     0.5754973  0.3632460
# 3     0.5701365  0.3552046
# 5     0.5509981  0.3264973
# 6     0.5480644  0.3220967
# 8     0.5470866  0.3206299
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 2.

# Evaluate Model Performance on Validation Set
val_predictions <- predict(rf_model, newdata = val_data)
val_conf_matrix <- confusionMatrix(val_predictions, val_data$Outcome)
cat("Validation Confusion Matrix and Statistics:\n")
print(val_conf_matrix)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction At Risk Critical Healthy
# At Risk      978     1304    4961
# Critical    1586     6188    3373
# Healthy     1526      801   24283
# 
# Overall Statistics
# 
# Accuracy : 0.6989          
# 95% CI : (0.6946, 0.7031)
# No Information Rate : 0.7248          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4108          
# 
# Mcnemar's Test P-Value : <2e-16          
# 
# Statistics by Class:
# 
#                      Class: At Risk Class: Critical Class: Healthy
# Sensitivity                 0.23912          0.7462         0.7445
# Specificity                 0.84686          0.8649         0.8121
# Pos Pred Value              0.13503          0.5551         0.9126
# Neg Pred Value              0.91758          0.9378         0.5468
# Prevalence                  0.09089          0.1843         0.7248
# Detection Rate              0.02173          0.1375         0.5396
# Detection Prevalence        0.16096          0.2477         0.5913
# Balanced Accuracy           0.54299          0.8055         0.7783

# Evaluate Model Performance on Test Set
test_predictions <- predict(rf_model, newdata = test_data)
test_conf_matrix <- confusionMatrix(test_predictions, test_data$Outcome)
cat("Test Confusion Matrix and Statistics:\n")
print(test_conf_matrix)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction At Risk Critical Healthy
# At Risk      990     1223    5073
# Critical    1607     6206    3383
# Healthy     1493      863   24161
# 
# Overall Statistics
# 
# Accuracy : 0.6968          
# 95% CI : (0.6926, 0.7011)
# No Information Rate : 0.7248          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4082          
# 
# Mcnemar's Test P-Value : <2e-16          
# 
# Statistics by Class:
# 
#                      Class: At Risk Class: Critical Class: Healthy
# Sensitivity                 0.24205          0.7484         0.7407
# Specificity                 0.84610          0.8641         0.8097
# Pos Pred Value              0.13588          0.5543         0.9112
# Neg Pred Value              0.91780          0.9383         0.5425
# Prevalence                  0.09089          0.1843         0.7248
# Detection Rate              0.02200          0.1379         0.5369
# Detection Prevalence        0.16191          0.2488         0.5893
# Balanced Accuracy           0.54408          0.8062         0.7752


# Analyze Feature Importance from the Random Forest Model
cat("Feature Importance in the Random Forest Model:\n")
print(importance(rf_model$finalModel))  # Feature importance based on the Random Forest model

# MeanDecreaseGini
# Age                               1024.46199
# CholesterolLevel_YeoJohnson       2207.27983
# HealthcareCost_YeoJohnson         7829.22981
# ExerciseFrequencyOften              83.63021
# ExerciseFrequencyRarely             87.49264
# ExerciseFrequencySometimes          92.28537
# DiabetesYes                        100.81673
# HeartDiseaseYes                    100.44282





